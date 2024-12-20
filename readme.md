# Adaptive Reasoning with Tree of Thoughts (ToT)

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/amins01/adaptive-tree-of-thoughts-reasoning.git
    cd adaptive-tree-of-thoughts-reasoning
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

    -   Create a `.env` file in the root directory.
    -   Add your Google API key and MLflow tracking URI to the `.env` file:

    ```
    GOOGLE_API_KEY=<your_google_api_key>
    MLFLOW_TRACKING_URI=<your_mlflow_tracking_uri>
    ```

## Usage

 Experiment runs, metrics, and visualizations are logged to MLflow. You can view the results by starting the MLflow UI:

    ```bash
    mlflow ui
    ```

Run the `main.py` script:

```bash
python main.py \
    <initial_prompt> \
    <initial_reasoning_step_prompt_template_path> \
    <reasoning_step_prompt_template_path> \
    <final_reasoning_step_prompt_template_path> \
    <resource_allocation_prompt_template_path> \
    <reward_prompt_template_path> \
    <direct_baseline_prompt_template> \
    <cot_baseline_prompt_template>
```

The final reasoning will be outputted in the console.

## Visualization

After every thinking step, an interactive visualization of the ToT will be displayed. Once the reasoning is over, the final ToT will be displayed highlighting the best reasoning branch:
![image](https://github.com/user-attachments/assets/9666d3a5-1384-4f2d-9530-c166c9fc11d1)

