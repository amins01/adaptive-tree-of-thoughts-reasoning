import os
import argparse
import mlflow
from dotenv import load_dotenv

from models.llm import LLM
from tot.tree_of_thoughts import ToT

load_dotenv("./.env")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_prompt", help="Initial prompt")
    parser.add_argument("initial_reasoning_step_prompt_template_path", help="Path to the initial reasoning step prompt template")
    parser.add_argument("reasoning_step_prompt_template_path", help="Path to the reasoning step prompt template")
    parser.add_argument("final_reasoning_step_prompt_template_path", help="Path to the final reasoning step prompt template")
    parser.add_argument("resource_allocation_prompt_template_path", help="Path to the resource allocation prompt template")
    parser.add_argument("reward_prompt_template_path", help="Path to the reward prompt template")
    parser.add_argument("direct_baseline_prompt_template", help="Path to the direct baseline prompt template")
    parser.add_argument("cot_baseline_prompt_template", help="Path to the CoT baseline prompt template")
    # parser.add_argument("num_initial_reasoning_steps", help="Number of initial reasoning steps (k)")
    # parser.add_argument("max_resources_per_step", help="Maximum number of nodes to allocation per step")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with mlflow.start_run():
        reasoning_model_id = "gemini-1.5-flash" # tunedModels/gemini15flash001tuningv1-iyspaq3lkhye
        model_id = "gemini-1.5-flash"
        
        reasoning_policy_model = LLM(model_id=reasoning_model_id) # "meta-llama/Llama-3.1-8B-Instruct" , temperature=2, top_p=1
        resource_allocation_policy_model = LLM(model_id=model_id, temperature=0.5)
        reward_model = LLM(model_id=model_id, temperature=0.5)

        tot = ToT(
            reasoning_policy_model=reasoning_policy_model,
            resource_allocation_policy_model=resource_allocation_policy_model,
            reward_model=reward_model,
            initial_reasoning_step_prompt_template_path=args.initial_reasoning_step_prompt_template_path,
            reasoning_step_prompt_template_path=args.reasoning_step_prompt_template_path,
            final_reasoning_step_prompt_template_path=args.final_reasoning_step_prompt_template_path,
            resource_allocation_prompt_template_path=args.resource_allocation_prompt_template_path,
            reward_prompt_template_path=args.reward_prompt_template_path
        )

        reasoning, final_answer = tot.think(
            initial_prompt=args.initial_prompt,
            num_initial_reasoning_steps=4, # k
            max_resources_per_step=4, # m
            sampling_method="epsilon-greedy",
            final_answer_sampling_method="greedy",
            num_samples_temperature=0.8, # T
            sampling_temperature=0.5,
            epsilon=0.2,
            min_epsilon=0.01,
            epsilon_decay_rate=0.99,
            display_tree=True
        )

        print(f"Tree - Final reasoning: {reasoning}")
        print(f"Tree - Final answer: {final_answer}")

        # TODO: adjust
        baseline_model = LLM(model_id=model_id)

        with open(args.direct_baseline_prompt_template, "r") as direct_baseline_f,\
             open(args.cot_baseline_prompt_template, "r") as cot_baseline_f:
            direct_baseline_prompt_template = direct_baseline_f.read()
            cot_baseline_prompt_template = cot_baseline_f.read()
        
        direct_baseline_prompt = direct_baseline_prompt_template.format(
            initial_prompt=args.initial_prompt
        )
        mlflow.log_text(direct_baseline_prompt, f"direct/direct_baseline_prompt.txt")

        direct_baseline_response, input_token_count, output_token_count = baseline_model.generate(direct_baseline_prompt)
        mlflow.log_text(direct_baseline_response, f"direct/direct_baseline_response.txt")
        mlflow.log_metric(f"direct_input_token_count", input_token_count)
        mlflow.log_metric(f"direct_output_token_count", output_token_count)
        mlflow.log_metric(f"direct_total_token_count", input_token_count + output_token_count)
        print(f"Direct (baseline) - Final answer: {direct_baseline_response}")

        cot_baseline_prompt = cot_baseline_prompt_template.format(
            initial_prompt=args.initial_prompt
        )
        mlflow.log_text(cot_baseline_prompt, f"cot/cot_baseline_prompt.txt")

        cot_baseline_response, input_token_count, output_token_count = baseline_model.generate(cot_baseline_prompt)
        mlflow.log_text(cot_baseline_response, f"cot/cot_baseline_response.txt")
        mlflow.log_metric(f"cot_input_token_count", input_token_count)
        mlflow.log_metric(f"cot_output_token_count", output_token_count)
        mlflow.log_metric(f"cot_total_token_count", input_token_count + output_token_count)
        print(f"CoT (baseline) - Final answer: {cot_baseline_response}")

if __name__ == "__main__":
    main()