"""
Command-line interface for Die Prompterei.
"""

import json
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from .core.models import TaskConfig
from .core.orchestrator import Orchestrator

console = Console()


@click.group()
def cli():
    """
    Die Prompterei - Prompt Optimization with Sacred Train/Test Separation.

    The Poet crafts prompts, the Judge evaluates them on hidden validation data,
    and the Guardian filters feedback to preserve the sacred separation.
    """
    pass


@cli.command()
@click.argument("task_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    type=click.Path(),
    default="receipts",
    help="Directory to save optimization receipts",
)
def optimize(task_file: str, output_dir: str):
    """
    Run prompt optimization for a task.

    TASK_FILE: Path to task configuration YAML file
    """
    console.print("[bold]Die Prompterei[/bold] - Starting optimization...\n")

    # Load task configuration
    with open(task_file, encoding="utf-8") as f:
        task_data = yaml.safe_load(f)

    task_config = TaskConfig(**task_data)
    console.print(f"Task: [cyan]{task_config.task.name}[/cyan]")
    console.print(f"Type: [cyan]{task_config.task.type}[/cyan]")
    console.print(f"Goal: {task_config.task.goal}\n")

    # Run optimization
    try:
        orchestrator = Orchestrator(task_config)
        receipt = orchestrator.optimize()

        # Display results
        console.print("\n[bold green]Optimization Complete![/bold green]\n")

        # Create results table
        table = Table(title="Optimization Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Final Score", f"{receipt.final_score:.4f}")
        table.add_row("Rounds", str(len(receipt.rounds)))
        table.add_row("Convergence", receipt.convergence_reason)
        table.add_row("API Calls", str(receipt.total_cost["api_calls"]))
        table.add_row(
            "Tokens",
            f"{receipt.total_cost['tokens_input']} in / {receipt.total_cost['tokens_output']} out",
        )
        table.add_row("Duration", f"{receipt.total_cost['duration_seconds']:.1f}s")

        console.print(table)

        # Save receipt
        output_path = Path(output_dir)
        orchestrator.save_receipt(receipt, output_path)

        console.print(
            f"\n[bold]Results saved to:[/bold] {output_path / task_config.task.name}_best.txt"
        )
        console.print(
            f"[bold]Full receipt:[/bold] {output_path / f'{task_config.task.name}_{receipt.timestamp}.json'}"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


@cli.command()
@click.argument("task_file", type=click.Path(exists=True))
@click.option("--prompt", required=True, type=click.Path(exists=True), help="Path to prompt file")
def eval(task_file: str, prompt: str):
    """
    Evaluate a specific prompt against validation data.

    TASK_FILE: Path to task configuration YAML file
    """
    console.print("[bold]Die Prompterei[/bold] - Evaluating prompt...\n")

    # Load task configuration
    with open(task_file, encoding="utf-8") as f:
        task_data = yaml.safe_load(f)

    task_config = TaskConfig(**task_data)

    # Load prompt
    with open(prompt, encoding="utf-8") as f:
        prompt_text = f.read()

    # Evaluate
    try:
        from .core.llm_client import create_llm_client

        orchestrator = Orchestrator(task_config)
        execution_client = create_llm_client(task_config.task.llm)

        result = orchestrator._evaluate_prompt(prompt_text, execution_client)

        # Display results
        console.print("[bold green]Evaluation Complete![/bold green]\n")

        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Primary Score", f"{result['judge_output'].scores.primary:.4f}")
        table.add_row("Variance", f"{result['judge_output'].scores.variance:.4f}")
        table.add_row("Final Score", f"{result['final_score']:.4f}")

        for metric, value in result["judge_output"].scores.secondary.items():
            table.add_row(f"  {metric}", f"{value:.4f}")

        console.print(table)

        # Display error patterns
        if result["guardian_output"].error_patterns:
            console.print("\n[bold]Error Patterns:[/bold]")
            for pattern in result["guardian_output"].error_patterns:
                console.print(f"  • {pattern}")

        # Display suggestions
        if result["guardian_output"].suggestions:
            console.print("\n[bold]Suggestions:[/bold]")
            for suggestion in result["guardian_output"].suggestions:
                console.print(f"  • {suggestion}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


@cli.command()
@click.argument("task_name")
@click.option(
    "--receipts-dir",
    type=click.Path(exists=True),
    default="receipts",
    help="Directory containing receipts",
)
def history(task_name: str, receipts_dir: str):
    """
    View optimization history for a task.

    TASK_NAME: Name of the task to view history for
    """
    receipts_path = Path(receipts_dir)

    # Find all receipts for this task
    receipt_files = list(receipts_path.glob(f"{task_name}_*.json"))

    if not receipt_files:
        console.print(f"[yellow]No receipts found for task '{task_name}'[/yellow]")
        return

    console.print(f"[bold]Optimization History for {task_name}[/bold]\n")

    # Load and display each receipt
    for receipt_file in sorted(receipt_files):
        with receipt_file.open("r", encoding="utf-8") as f:
            receipt_data = json.load(f)

        console.print(f"[cyan]Run: {receipt_data['timestamp']}[/cyan]")
        console.print(f"  Final Score: {receipt_data['final_score']:.4f}")
        console.print(f"  Rounds: {len(receipt_data['rounds'])}")
        console.print(f"  Convergence: {receipt_data['convergence_reason']}")
        console.print(f"  Duration: {receipt_data['total_cost']['duration_seconds']:.1f}s\n")


@cli.command()
@click.argument("task_name")
@click.option(
    "--receipts-dir",
    type=click.Path(exists=True),
    default="receipts",
    help="Directory containing receipts",
)
@click.option("--output", required=True, type=click.Path(), help="Output file path")
def export(task_name: str, receipts_dir: str, output: str):
    """
    Export the best prompt for a task.

    TASK_NAME: Name of the task
    """
    best_prompt_file = Path(receipts_dir) / f"{task_name}_best.txt"

    if not best_prompt_file.exists():
        console.print(
            f"[yellow]No optimized prompt found for task '{task_name}'[/yellow]"
        )
        console.print("Run 'prompterei optimize' first.")
        return

    # Copy best prompt to output location
    with best_prompt_file.open("r", encoding="utf-8") as f:
        prompt = f.read()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(prompt)

    console.print(f"[green]Best prompt exported to:[/green] {output_path}")


if __name__ == "__main__":
    cli()
