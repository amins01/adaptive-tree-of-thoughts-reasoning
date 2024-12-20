import numpy as np

class ThoughtNode():
    def __init__(
        self,
        thought: str | None,
        generation_prompt: str | None,
        id: int
    ):
        self.thought = thought
        self.generation_prompt = generation_prompt
        self.id: int = id
        self.tree_level: int = 0
        self.quality_score: float = -1
        self.next_step_allocated_resources: int = -1
        self.efficiency_score: float = -1
        self.combined_reward: float = -1
        self.parent: ThoughtNode | None = None
        self.pruned: bool = False
        self.winner: bool = False
        self.prev_reasoning_path = []
        self.children: list[ThoughtNode] = []

    def add_child(self, node):
        node.parent = self
        node.prev_reasoning_path = self.get_full_reasoning_path()
        node.tree_level = self.tree_level + 1
        self.children.append(node)

    def update_next_step_allocated_resources(self, allocated_resources: int):
        self.next_step_allocated_resources = int(allocated_resources)
        self.efficiency_score = 1 / (np.log2(allocated_resources + 2))

    def update_combined_reward(self, qs_weight: float, es_weight: float):
        # TODO: return instead of updating class var
        self.combined_reward = qs_weight * self.quality_score + es_weight * self.efficiency_score

    def set_winner_branch(self):
        if self.tree_level == 0:
            return
        
        current_node = self

        while current_node.tree_level > 0:
            current_node.winner = True
            current_node = current_node.parent

    def get_cumulative_reward(self):
        # TODO: clean up

        if self.tree_level == 0:
            return -1
        
        cumulative_reward = self.combined_reward
        current_node = self.parent

        while current_node.tree_level > 0:
            cumulative_reward += current_node.combined_reward
            current_node = current_node.parent
        
        return cumulative_reward / self.tree_level

    def get_full_reasoning_path(self):
        reasoning_path_copy = self.prev_reasoning_path.copy()

        if self.tree_level > 0:
            reasoning_path_copy.append(self.thought)
        
        return reasoning_path_copy
    
    def get_siblings(self):
        siblings = []

        if not self.parent:
            return siblings

        for child in self.parent.children:
            if child.id != self.id:
                siblings.append(child)

        return siblings

    def is_final_thought(self):
        return not bool(self.next_step_allocated_resources)
