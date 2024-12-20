import json
import logging
import mlflow
import numpy as np
from tqdm import tqdm
import random

from models.llm import LLM
from tot.thought_node import ThoughtNode
from utils.processing import (
    get_reasoning_path_string,
    preprocess_reasoning_step_response,
    preprocess_resource_allocation_response,
    preprocess_reward_response,
    get_sibling_thoughts_string,
    save_branch_for_finetuning,
    save_input_output_for_finetuning
)
from utils.tracking import log_thought_node

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ToT():
    """
    Class responsible for the adapative ToT.
    """
    def __init__(
        self,
        reasoning_policy_model: LLM,
        resource_allocation_policy_model: LLM,
        reward_model: LLM,
        initial_reasoning_step_prompt_template_path,
        reasoning_step_prompt_template_path,
        final_reasoning_step_prompt_template_path,
        resource_allocation_prompt_template_path,
        reward_prompt_template_path,
        qs_weight=1,
        es_weight=1,
        traversal_algo="bfs",
    ):
        self.reasoning_policy_model = reasoning_policy_model
        self.resource_allocation_policy_model = resource_allocation_policy_model
        self.reward_model = reward_model
        self.initial_reasoning_step_prompt_template_path = initial_reasoning_step_prompt_template_path
        self.reasoning_step_prompt_template_path = reasoning_step_prompt_template_path
        self.final_reasoning_step_prompt_template_path = final_reasoning_step_prompt_template_path
        self.resource_allocation_prompt_template_path = resource_allocation_prompt_template_path
        self.reward_prompt_template_path = reward_prompt_template_path
        self.qs_weight: float = qs_weight
        self.es_weight: float = es_weight
        self.traversal_algo: str = traversal_algo
        self.input_token_count: int = 0
        self.output_token_count: int = 0
        self.node_count: int = 0
        self.all_nodes: list[ThoughtNode] = []
        self.nodes_to_explore: list[ThoughtNode] = []
        self.final_thoughts: list[ThoughtNode] = []
        self._init_prompt_templates()

    def _init_prompt_templates(self):
        with open(self.reward_prompt_template_path, "r") as reward_f,\
            open(self.resource_allocation_prompt_template_path, "r") as resource_allocation_f,\
            open(self.initial_reasoning_step_prompt_template_path, "r") as initial_reasoning_step_f,\
            open(self.reasoning_step_prompt_template_path, "r") as reasoning_step_f,\
            open(self.final_reasoning_step_prompt_template_path, "r") as final_reasoning_step_f:
            self.reward_prompt_template = reward_f.read()
            self.resource_allocation_prompt_template = resource_allocation_f.read()
            self.initial_reasoning_step_prompt_template = initial_reasoning_step_f.read()
            self.reasoning_step_prompt_template = reasoning_step_f.read()
            self.final_reasoning_step_prompt_template = final_reasoning_step_f.read()
    
    def _init_tot(self, initial_prompt, num_initial_reasoning_steps, logging_id):
        """
        """
        logger.info(f"========== Generating initial thoughts ==========")

        # TODO: assign None as thought?
        root_node = self._create_thought_node(thought=initial_prompt, generation_prompt=None)

        for _ in range(num_initial_reasoning_steps):
            initial_reasoning_step_prompt = self.initial_reasoning_step_prompt_template.format(
                initial_prompt=initial_prompt,
                sibling_thoughts=get_sibling_thoughts_string(root_node.children, id_prefix=False) if root_node.children else "None"
            )

            mlflow.log_text(initial_reasoning_step_prompt, f"{logging_id}/reasoning/initial_reasoning_prompt_node_{self.node_count + 1}.txt")
            response, input_token_count, output_token_count = self.reasoning_policy_model.generate(initial_reasoning_step_prompt)
            node = self._create_thought_node(thought=response, generation_prompt=initial_reasoning_step_prompt)
            root_node.add_child(node)
            self.input_token_count += input_token_count
            self.output_token_count += output_token_count
            mlflow.log_text(response, f"{logging_id}/reasoning/initial_reasoning_response_node_{node.id}.txt")

    def _create_thought_node(self, thought, generation_prompt):
        self.node_count += 1
        thought_node = ThoughtNode(
            thought=thought,
            generation_prompt=generation_prompt,
            id=self.node_count
        )
        self.all_nodes.append(thought_node)
        self.nodes_to_explore.append(thought_node)

        return thought_node

    def _get_adaptive_num_samples(self, scores, max_samples=3, temperature=0.5):
        """Calculates adaptive number of samples using softmax."""
        if not scores:
            return 1

        highest_score = max(scores)
        score_differences = np.array([highest_score - s for s in scores])

        # Softmax with temperature
        probs = np.exp(-score_differences / temperature) / np.sum(np.exp(-score_differences / temperature))

        # Sample the number of samples using calculated distribution (without replacement, 1 at a time)
        num_samples = 0
        for i in range(0, min(len(scores), max_samples)):
            rand = random.random()
            if rand < probs[i]:
                num_samples += 1

        return num_samples if num_samples else 1

    def _sample_greedy(self, items: list, values: list):
        max_index = values.index(max(values))
        sampled_items = [items[max_index]]
        return sampled_items

    def _sample_random(self, items: list, values: list = None, num_samples=None):
        count = len(items)

        if count == 0:
            return []

        sample_size = num_samples if num_samples else random.randint(1, count)
        sampled_indices = random.sample(range(count), sample_size)
        # TODO: shorten
        sampled_items = [items[i] for i in sampled_indices] if values is None else [values[i] for i in sampled_indices]
        return sampled_items

    def _sample_epsilon_greedy(self, items: list, values: list, num_samples_temperature: float):
        num_samples = self._get_adaptive_num_samples(
            scores=values,
            max_samples=len(values),
            temperature=num_samples_temperature
        )
        sampled_items = []

        if random.random() < self.epsilon: # Explore
            sampled_items = self._sample_random(items, num_samples=min(num_samples, len(items)))
        else: # Exploit
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
            sampled_indices = sorted_indices[:min(num_samples, len(items))]
            sampled_items = [items[i] for i in sampled_indices]

        return sampled_items

    def _sample_thoughts_temperature(self, items: list, values: list, sampling_temperature, num_samples_temperature):
        num_samples = self._get_adaptive_num_samples(
            scores=values,
            max_samples=len(values),
            temperature=num_samples_temperature
        )
        
        logits = np.array(values) / sampling_temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))

        try:
            selected_indices = np.random.choice(len(items), size=min(num_samples, len(items)), replace=False, p=probs)
            sampled_thoughts = [items[i] for i in selected_indices]
        except ValueError as e:
            # all probabilities are 0 (rounding)
            logger.warning(f"Error during temperature sampling: {e}. Defaulting to greedy sampling.")
            sampled_thoughts = self._sample_greedy(items, values)

        return sampled_thoughts

    def _sample_thoughts(
        self,
        thought_nodes: list[ThoughtNode],
        scores: list[float],
        sampling_method: str,
        num_samples_temperature: float,
        sampling_temperature: float,
        pruning: bool,
        logging_id: str | int,
    ):
        if sampling_method == "random":
            sampled_thoughts = self._sample_random(thought_nodes)
        elif sampling_method == "greedy":
            sampled_thoughts = self._sample_greedy(thought_nodes, scores)
        elif sampling_method == "epsilon-greedy":
            sampled_thoughts = self._sample_epsilon_greedy(
                thought_nodes,
                scores,
                num_samples_temperature
            )
        elif sampling_method == "temperature":
            sampled_thoughts = self._sample_thoughts_temperature(
                thought_nodes,
                scores,
                sampling_temperature,
                num_samples_temperature
            )
        else:
            logger.error(f"Unrecognized sampling method: {sampling_method}")
            return None
        
        if pruning:
            pruned_thoughts = [node for node in thought_nodes if node not in sampled_thoughts]
            for node in pruned_thoughts:
                node.pruned = True
                self.nodes_to_explore.remove(node)
                log_thought_node(node, logging_id)
        
        return sorted(sampled_thoughts, key=lambda t: t.id)

    def _generate_baseline_thoughts(self, current_node: ThoughtNode):
        baselines = []

        if current_node.tree_level > 0:
            duplicate = ThoughtNode(
                thought=current_node.parent.thought,
                generation_prompt=None,
                id=current_node.id
            )
            baselines.append(duplicate)

        # TODO: add correction as baseline

        no_op_thought = ThoughtNode(
            thought="Continue with the current approach.\n",
            generation_prompt=None,
            id=self.node_count + 1
        )

        baselines.append(no_op_thought)

        return baselines

    def _think_step(
        self,
        initial_prompt: str,
        current_node: ThoughtNode,
        max_resources_per_step: int,
        sampling_method: str,
        num_samples_temperature: float,
        sampling_temperature: float,
        pruning: bool,
        logging_id
    ):
        # evalute generated thoughts
        logger.info(f"Evaluating children thoughts of node {current_node.id}...")
        baseline_thoughts = self._generate_baseline_thoughts(current_node)
        thoughts_to_evaluate = baseline_thoughts + current_node.children
        thoughts_to_evaluate_str = get_sibling_thoughts_string(random.sample(thoughts_to_evaluate, len(thoughts_to_evaluate)))
        # TODO: separate thoughts clearly
        reward_prompt = self.reward_prompt_template.format(
            initial_prompt=initial_prompt,
            reasoning_path=get_reasoning_path_string(current_node.get_full_reasoning_path()),
            sibling_thoughts=thoughts_to_evaluate_str,
        )
        mlflow.log_text(reward_prompt, f"{logging_id}/rewards/reward_prompt_node_{current_node.id}.txt")
        reward_response, input_token_count, output_token_count = self.reward_model.generate(reward_prompt)
        reward_response = preprocess_reward_response(reward_response)
        self.input_token_count += input_token_count
        self.output_token_count += output_token_count
        mlflow.log_text(str(reward_response), f"{logging_id}/rewards/reward_response_node_{current_node.id}.txt")
        self._update_siblings_quality_score(reward_response, current_node.children, baseline_thoughts)

        # determine budget for each nodes
        logger.info(f"Allocating resources for children thoughts of node {current_node.id}...")
        for child in current_node.children:
            resource_allocation_prompt = self.resource_allocation_prompt_template.format(
                initial_prompt=initial_prompt,
                reasoning_path=get_reasoning_path_string(child.prev_reasoning_path),
                current_thought=child.thought,
                max_resources_per_step=max_resources_per_step,
            )
            mlflow.log_text(resource_allocation_prompt, f"{logging_id}/resource_allocation/resource_allocation_prompt_node_{current_node.id}_{child.id}.txt")
            resource_allocation_response, input_token_count, output_token_count = self.resource_allocation_policy_model.generate(resource_allocation_prompt)
            resource_allocation_response = preprocess_resource_allocation_response(resource_allocation_response)
            self.input_token_count += input_token_count
            self.output_token_count += output_token_count
            mlflow.log_text(str(resource_allocation_response), f"{logging_id}/resource_allocation/resource_allocation_response_node_{current_node.id}_{child.id}.txt")
            child.update_next_step_allocated_resources(resource_allocation_response)
            child.update_combined_reward(self.qs_weight, self.es_weight)
        
        # filter thoughts (sample thoughts based on QS and ES)
        logger.info(f"Sampling (based on QS and ES) children thoughts of node {current_node.id}...")
        mlflow.log_text(str([n.id for n in current_node.children]), f"{logging_id}/sampling/children_nodes_all_node_{current_node.id}.txt")
        combined_rewards = [n.combined_reward for n in current_node.children]
        sampled_thoughts = self._sample_thoughts(
            thought_nodes=current_node.children,
            scores=combined_rewards,
            sampling_method=sampling_method,
            num_samples_temperature=num_samples_temperature,
            sampling_temperature=sampling_temperature,
            pruning=pruning,
            logging_id=logging_id
        )
        mlflow.log_text(str([n.id for n in sampled_thoughts]), f"{logging_id}/sampling/children_nodes_sampled_node_{current_node.id}.txt")
        
        # use the budget to generate next thoughts
        for thought_node in sampled_thoughts:
            logger.info(f"Generate next thoughts (based on allocated resources) from node {thought_node.id}...")
            if thought_node.is_final_thought():
                self.nodes_to_explore.remove(thought_node)
                self.final_thoughts.append(thought_node)
                log_thought_node(thought_node, logging_id)
            
            for i in range(thought_node.next_step_allocated_resources):
                # TODO: separate thoughts clearly
                reasoning_step_prompt = self.reasoning_step_prompt_template.format(
                    initial_prompt=initial_prompt,
                    reasoning_path=get_reasoning_path_string(thought_node.prev_reasoning_path),
                    current_thought=thought_node.thought,
                    sibling_thoughts=get_sibling_thoughts_string(thought_node.children, id_prefix=False) if thought_node.children else "None",
                    max_resources_per_step=max_resources_per_step
                )
                mlflow.log_text(reasoning_step_prompt, f"{logging_id}/reasoning/reasoning_prompt_parent_node_{thought_node.id}_{i}.txt")
                reasoning_step_response, input_token_count, output_token_count = self.reasoning_policy_model.generate(reasoning_step_prompt)
                reasoning_step_response = preprocess_reasoning_step_response(reasoning_step_response)
                self.input_token_count += input_token_count
                self.output_token_count += output_token_count
                mlflow.log_text(reasoning_step_response, f"{logging_id}/reasoning/reasoning_response_parent_node_{thought_node.id}_{i}.txt")
                child_node = self._create_thought_node(thought=reasoning_step_response, generation_prompt=reasoning_step_prompt)
                thought_node.add_child(child_node)

    def _get_next_explorable_node(self):
        if not self.nodes_to_explore:
            logger.warning("No more nodes to explore")
            return None

        if self.traversal_algo == "bfs":
            return self.nodes_to_explore.pop(0)
        
        if self.traversal_algo == "dfs":
            return self.nodes_to_explore.pop()
        
        logger.error(f"Invalid traversal algorithm: {self.traversal_algo}")
        return None

    def _update_siblings_quality_score(self, ranking_list, siblings: list[ThoughtNode], baseline_thoughts: list[ThoughtNode]):
         # TODO: move/use another function?
        all_thoughts = siblings + baseline_thoughts
        
        count = len(ranking_list)
        quality_scores = {}

        # Squared scoring
        for i in range(count):
            quality_scores[ranking_list[i]] = (count - i)**2

        # Normalize
        max_score = max(quality_scores.values()) if quality_scores else 1
        for thought_id in quality_scores:
            quality_scores[thought_id] /= max_score

        for s_node in all_thoughts:
            if s_node.id in quality_scores:
                s_node.quality_score = quality_scores[s_node.id]
            else:
                s_node.quality_score = 0
                logger.error(f"ID {s_node.id} not in ranking. Check reward llm output (probably caused by duplicated sibling thoughts)")

    def _generate_final_answer(
        self,
        initial_prompt: str,
        sampling_method: str,
        num_samples_temperature: float,
        sampling_temperature: float,
        logging_id: str | int,
        ft_dataset_path: str | None
    ):
        # TODO: allow different methods?
        cumulative_rewards = [t.get_cumulative_reward() for t in self.final_thoughts]
        sampled_thought = self._sample_thoughts(
            thought_nodes=self.final_thoughts,
            scores=cumulative_rewards,
            sampling_method=sampling_method,
            num_samples_temperature=num_samples_temperature,
            sampling_temperature=sampling_temperature,
            pruning=False,
            logging_id=logging_id
        )[0]
        sampled_thought.set_winner_branch()

        full_reasoning_path = sampled_thought.get_full_reasoning_path()
        mlflow.log_text(str(full_reasoning_path), f"{logging_id}/reasoning/final_full_reasoning_path_node_{sampled_thought.id}.txt")

        final_reasoning_step_prompt = self.final_reasoning_step_prompt_template.format(
            initial_prompt=initial_prompt,
            reasoning_path=get_reasoning_path_string(full_reasoning_path)
        )

        mlflow.log_text(final_reasoning_step_prompt, f"{logging_id}/reasoning/final_reasoning_prompt_node_{sampled_thought.id}.txt")
        final_reasoning_step_response, input_token_count, output_token_count = self.reasoning_policy_model.generate(final_reasoning_step_prompt)
        final_reasoning_step_response = preprocess_reasoning_step_response(final_reasoning_step_response)
        self.input_token_count += input_token_count
        self.output_token_count += output_token_count
        mlflow.log_text(final_reasoning_step_response, f"{logging_id}/reasoning/final_reasoning_response_node_{sampled_thought.id}.txt")

        if ft_dataset_path:
            save_branch_for_finetuning(self.final_thoughts, ft_dataset_path)
            save_input_output_for_finetuning(final_reasoning_step_prompt, final_reasoning_step_response, ft_dataset_path)
            mlflow.log_artifact(ft_dataset_path, f"{logging_id}/finetuning_dataset.jsonl")

        mlflow.log_metric(f"tot_input_token_count", self.input_token_count)
        mlflow.log_metric(f"tot_output_token_count", self.output_token_count)
        mlflow.log_metric(f"tot_total_token_count", self.input_token_count + self.output_token_count)

        return full_reasoning_path, final_reasoning_step_response

    def think(
        self,
        initial_prompt: str,
        num_initial_reasoning_steps: int,
        max_resources_per_step: int,
        sampling_method: str,
        final_answer_sampling_method: str,
        num_samples_temperature: float,
        sampling_temperature: float,
        epsilon: float,
        min_epsilon: float,
        epsilon_decay_rate: float,
        display_tree: bool = False,
        ft_dataset_path: str | None = None,
        logging_id="."
    ):
        """
        """
        print(initial_prompt) # TODO: remove
        mlflow.log_text(initial_prompt, f"{logging_id}/initial_prompt.txt")
        self.node_count = 0
        self.input_token_count = 0
        self.output_token_count = 0
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.all_nodes = []
        self.nodes_to_explore = []
        self.final_thoughts = []
        self._init_tot(initial_prompt, num_initial_reasoning_steps, logging_id)

        from utils.visualization import generate_tree_visualization

        while self.nodes_to_explore:
            logger.info(f"========== Nodes to explore: {[n.id if not n.pruned else -1 for n in self.nodes_to_explore]} ==========")
            current_node = self._get_next_explorable_node()
            logger.info(f"========== Exploring thought node {current_node.id} ==========")

            self._think_step(
                initial_prompt=initial_prompt,
                current_node=current_node,
                max_resources_per_step=max_resources_per_step,
                sampling_method=sampling_method,
                num_samples_temperature=num_samples_temperature,
                sampling_temperature=sampling_temperature,
                pruning=True,
                logging_id=logging_id
            )
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)
            
            log_thought_node(current_node, logging_id)
            generate_tree_visualization(self, display=display_tree, logging_id=logging_id) # TODO: remove
        
        full_reasoning_path, final_reasoning_step_response = self._generate_final_answer(
            initial_prompt,
            sampling_method=final_answer_sampling_method,
            num_samples_temperature=num_samples_temperature,
            sampling_temperature=sampling_temperature,
            logging_id=logging_id,
            ft_dataset_path=ft_dataset_path
        )
 
        generate_tree_visualization(self, display=display_tree, logging_id=logging_id)
        
        return full_reasoning_path, final_reasoning_step_response