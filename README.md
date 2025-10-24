# Die Prompterei

<img width="1024" height="768" alt="image" src="https://github.com/user-attachments/assets/9eb9eb1a-df85-47df-a673-d97edabb1116" />

## The Mystical Workshop of Prompt Perfection

Welcome to Die Prompterei - a sacred guild hall where the ancient art of creation meets the rigorous science of validation. Like the traditional craftsman's workshop where master and apprentice labor side by side, this is where AI agents collaborate in the eternal dance of artistic creation and critical judgment.

## The Philosophy

In the spirit of five-thousand-year-old wisdom, we understand that true mastery comes not from solitary genius, but from the harmony between creation (yang) and evaluation (yin). Just as traditional machine learning has long honored the sacred principle of hold-out test data - keeping validation separate from training - Die Prompterei embodies this wisdom in the age of large language models.

## The Sacred Process

Within our digital promptelier, three AI spirits work in eternal collaboration:

### Der Dichter (The Poet)
- Crafts prompts with passion and creativity
- Draws from vast wells of artistic knowledge
- Remains innocent of the final judgment
- Iterates based on wisdom received from the shadows

### Der Richter (The Judge)
- Tests prompts against sacred validation data
- Sees what the Poet cannot see
- Performs detailed error analysis
- Ensures the ancient law: "Never shall the creator see the test"

### Der Wächter (The Guardian)
- Filters feedback to prevent data leakage
- Ensures validation data never reaches the Poet
- Transforms specific errors into aggregated patterns
- Maintains the sanctity of train/test separation

## The Workshop Flow

```
1. Poet crafts → 2. Judge evaluates → 3. Guardian filters → 4. Poet refines
                     ↑                        ↓
               [Validation Data]      [Sanitized Feedback]
```

## Why This Matters

In an age where LLMs can memorize and overfit, Die Prompterei resurrects the ancient wisdom:

- True validation requires separation
- The creator must not see the test data
- Feedback must be filtered through wisdom
- Iteration without contamination

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DiePrompterei.git
cd DiePrompterei

# Install dependencies with uv
uv sync

# Set your API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Run Your First Optimization

```bash
# Optimize a sentiment classification prompt
uv run python -m dieprompterei.cli optimize tasks/examples/classification.yaml

# View the results
cat receipts/sentiment_classification_best.txt
```

### Example Output

```
Die Prompterei - Starting optimization...

Task: sentiment_classification
Type: classification
Goal: Classify customer reviews into positive, negative, or neutral sentiment

Optimization Complete!

╭─────────────────────────────────────────────╮
│ Metric          │ Value                    │
├─────────────────────────────────────────────┤
│ Final Score     │ 0.9200                   │
│ Rounds          │ 4                        │
│ Convergence     │ improvement_below_thresh │
│ API Calls       │ 45                       │
│ Tokens          │ 12000 in / 3500 out     │
│ Duration        │ 42.3s                    │
╰─────────────────────────────────────────────╯

Results saved to: receipts/sentiment_classification_best.txt
```

---

## Task Configuration

Define tasks using YAML configuration files. See [tasks/examples/](tasks/examples/) for complete examples.

### Basic Structure

```yaml
task:
  name: "my_task"
  type: "classification"  # classification|extraction|qa|generation
  goal: "Your task description"

  output_schema:
    type: "object"
    properties:
      # JSON schema for expected output

  validation:
    path: "./data/my_validation.jsonl"

  metrics:
    primary: "exact_match"  # Primary metric to optimize
    secondary: ["precision", "recall"]

  scoring:
    base_metric_weight: 1.0
    variance_penalty_weight: 0.2  # Penalize inconsistent prompts

  optimization:
    max_rounds: 6
    candidates_per_round: 4
    poet_temperature: 0.5
    convergence:
      improvement_threshold: 0.02  # Stop if improvement < 2%
      plateau_rounds: 2            # Stop after N rounds without improvement
      min_rounds: 3                # Always run at least this many

  llm:
    provider: "anthropic"  # anthropic|openai
    model: "claude-sonnet-4-5"  # or "gpt-4" for OpenAI
    judge_temperature: 0.2
    guardian_temperature: 0.3

  baseline_prompt: |
    Your initial prompt here with {input} placeholder
```

### Validation Data Format

Validation data is provided as JSONL (one JSON object per line):

```jsonl
{"input": "example text", "expected": {"sentiment": "positive"}}
{"input": "another example", "expected": {"sentiment": "negative"}}
```

---

## CLI Commands

### Optimize a Task

```bash
uv run python -m dieprompterei.cli optimize tasks/examples/classification.yaml
```

Runs the full optimization loop and saves results to `receipts/`.

### Evaluate a Specific Prompt

```bash
uv run python -m dieprompterei.cli eval tasks/examples/classification.yaml --prompt my_prompt.txt
```

Tests a specific prompt against validation data.

### View Optimization History

```bash
uv run python -m dieprompterei.cli history sentiment_classification
```

Shows all optimization runs for a task.

### Export Best Prompt

```bash
uv run python -m dieprompterei.cli export sentiment_classification --output production_prompt.txt
```

Exports the best optimized prompt for production use.

---

## Supported Task Types

### Classification

Classify inputs into predefined categories.

**Metrics:** exact_match, f1_macro, f1_micro, precision, recall

**Example:** [tasks/examples/classification.yaml](tasks/examples/classification.yaml)

### Extraction

Extract structured entities or information from text.

**Metrics:** exact_match, span_f1, span_precision, span_recall

**Example:** [tasks/examples/extraction.yaml](tasks/examples/extraction.yaml)

### Question Answering

Answer questions based on provided context.

**Metrics:** exact_match, token_f1, token_precision, token_recall

**Example:** [tasks/examples/qa.yaml](tasks/examples/qa.yaml)

### Generation

Generate free-form text responses.

**Metrics:** token_f1, token_precision, token_recall

---

## Architecture

### Core Principles

1. **Sacred Separation**: The Poet never sees validation data
2. **Filtered Feedback**: Guardian sanitizes Judge's analysis
3. **Iterative Improvement**: Each round uses filtered feedback to generate better candidates
4. **Convergence Detection**: Stops when improvement plateaus or threshold is reached

### Component Responsibilities

- **Poet** (`core/poet.py`): Generates prompt candidates based on filtered feedback
- **Judge** (`core/judge.py`): Evaluates prompts on validation data, performs error analysis
- **Guardian** (`core/guardian.py`): Filters feedback to prevent validation data leakage
- **Orchestrator** (`core/orchestrator.py`): Coordinates the optimization loop

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator                           │
│                                                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│  │  Poet    │────▶│  Judge   │────▶│ Guardian │          │
│  │          │     │          │     │          │          │
│  │ Generates│     │Evaluates │     │ Filters  │          │
│  │candidates│     │on val set│     │ feedback │          │
│  └──────────┘     └──────────┘     └──────────┘          │
│       ▲                                    │               │
│       │                                    │               │
│       └────────────────────────────────────┘               │
│            Sanitized feedback only                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Development

### Project Structure

```
die-prompterei/
├── src/dieprompterei/
│   ├── core/              # Core agents and orchestration
│   │   ├── models.py      # Pydantic data models
│   │   ├── llm_client.py  # Abstract LLM interface
│   │   ├── poet.py        # Poet agent
│   │   ├── judge.py       # Judge agent
│   │   ├── guardian.py    # Guardian agent
│   │   └── orchestrator.py
│   ├── adapters/          # Metrics and validation
│   │   ├── metrics.py
│   │   └── schema_validator.py
│   ├── templates/         # Jinja2 prompt templates
│   └── cli.py             # Command-line interface
├── tasks/examples/        # Example task configurations
├── data/                  # Validation datasets
├── receipts/              # Optimization logs and results
└── tests/                 # Unit tests
```

### Running Tests

```bash
uv run pytest
```

### Adding a New LLM Provider

Implement the `LLMClient` interface in `core/llm_client.py`:

```python
class MyProviderClient(LLMClient):
    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        # Your implementation
        pass
```

Then register it in `create_llm_client()`.

---

## Contributing

Contributions welcome! Please ensure:

1. All agents maintain the sacred separation (Poet never sees validation data)
2. New metrics are task-specific and properly registered
3. Code follows modern Python 3.13+ patterns (Pydantic, pathlib, type hints)
4. Tests demonstrate that validation data doesn't leak

---

## License

MIT License - See LICENSE file for details
