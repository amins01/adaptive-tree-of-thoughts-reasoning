import mlflow
import json

from tot.thought_node import ThoughtNode

def log_thought_node(node: ThoughtNode, save_folder_id="."):
    """
    """
    node_info = {
        "thought": node.thought,
        "generation_prompt": node.generation_prompt,
        "is_final_thought": node.is_final_thought(),
        "pruned": node.pruned,
        "tree_level": node.tree_level,
        "next_step_allocated_resources": node.next_step_allocated_resources,
        "children": [n.id for n in node.children],
        "children_count": len(node.children),
        "full_reasoning_path": node.get_full_reasoning_path(),
        "quality_score": node.quality_score,
        "efficiency_score": node.efficiency_score,
        "combined_reward": node.combined_reward,
        "cumulative_reward": node.get_cumulative_reward()
    }
    json_content = json.dumps(node_info, indent=4)
    mlflow.log_text(json_content, f"{save_folder_id}/nodes/thought_node_{node.id}.json")

    if node.tree_level > 0:
        mlflow.log_metric("quality_score", node_info["quality_score"], step=node.id)
        mlflow.log_metric("efficiency_score", node_info["efficiency_score"], step=node.id)
        mlflow.log_metric("combined_reward", node_info["combined_reward"], step=node.id)
        mlflow.log_metric("cumulative_reward", node_info["cumulative_reward"], step=node.id)

def log_benchmark_results(method_name, prompt, response, correct, ground_truth, step_id):
    """
    """
    mlflow.log_text(prompt, f"{step_id}/benchmark_results/{method_name}_prompt.txt")
    mlflow.log_text(response, f"{step_id}/benchmark_results/{method_name}_response.txt")
    mlflow.log_metric(f"{method_name}_correct", int(correct), step=step_id)
    mlflow.log_text(f"Ground Truth: {ground_truth}", f"{step_id}/benchmark_results/{method_name}_ground_truth.txt")