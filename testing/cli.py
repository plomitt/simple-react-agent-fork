#!/usr/bin/env python3
"""
Command Line Interface for Testing System

Usage:
    poetry run python testing/cli.py list_configs_sorted
    poetry run python testing/cli.py list_runs_sorted
    poetry run python testing/cli.py list_summary
    poetry run python testing/cli.py --restart-searxng --agent-config-id <config_id>
    poetry run python testing/cli.py --restart-searxng --containers-to-restart redis,searxng,caddy,nginx
"""

import argparse
import os
from testing import run_complete_test


def main():
    parser = argparse.ArgumentParser(description='Testing System CLI')

    # Basic arguments (with defaults)
    parser.add_argument('--questions-file', default='testing/questions/questions.jsonl',
                       help='Path to questions JSONL file')
    parser.add_argument('--use-lmstudio', action='store_true', default=True,
                       help='Use LM Studio (default: True)')
    parser.add_argument('--max-steps', type=int, default=10,
                       help='Maximum steps for agent')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of questions to test')
    parser.add_argument('--output-dir', default='testing/results',
                       help='Output directory for results')

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()

    action_group.add_argument('--list-configs', action='store_true',
                             help='List all agent configurations')
    action_group.add_argument('--list-configs-sorted', action='store_true',
                             help='List configurations sorted by best score')
    action_group.add_argument('--list-runs', action='store_true',
                             help='List all available runs')
    action_group.add_argument('--list-runs-sorted', action='store_true',
                             help='List runs sorted by score')
    action_group.add_argument('--list-summary', action='store_true',
                             help='Show system performance summary')
    action_group.add_argument('--create-config', action='store_true',
                             help='Create a new configuration')

    # Configuration creation arguments
    parser.add_argument('--config-name', help='Name for new configuration')
    parser.add_argument('--config-model', help='Model for new configuration')
    parser.add_argument('--config-description', default='',
                       help='Description for new configuration')

    # Run arguments
    parser.add_argument('--resume', action='store_true',
                       help='Resume an incomplete run')
    parser.add_argument('--run-id', help='Specific run ID to resume')
    parser.add_argument('--cleanup-checkpoints', action='store_true',
                       help='Clean up checkpoint files after completion')
    parser.add_argument('--agent-config-id', help='Specific agent configuration ID to use')

    # Docker container arguments
    parser.add_argument('--restart-searxng', action='store_true', default=os.getenv('RESTART_SEARCH_CONTAINERS', '').lower() == 'true',
                       help='Restart search containers before each question to reset search limits (env: RESTART_SEARCH_CONTAINERS)')
    parser.add_argument('--containers-to-restart', default=os.getenv('CONTAINERS_TO_RESTART', 'redis,searxng,caddy'),
                       help='Comma-separated list of container names to restart (env: CONTAINERS_TO_RESTART, default: redis,searxng,caddy)')

    args = parser.parse_args()

    # Parse containers list
    containers_to_restart = None
    if args.restart_searxng:
        # Split comma-separated list and strip whitespace
        containers_to_restart = [c.strip() for c in args.containers_to_restart.split(',') if c.strip()]

    # Map CLI arguments to function parameters
    kwargs = {
        'questions_file': args.questions_file,
        'use_lmstudio': args.use_lmstudio,
        'max_steps': args.max_steps,
        'limit': args.limit,
        'show_validation_details': False,  # Default to False for CLI
        'output_dir': args.output_dir,
        'resume_mode': args.resume,
        'run_id': args.run_id,
        'list_runs': args.list_runs,
        'cleanup_checkpoints': args.cleanup_checkpoints,
        'agent_config_id': args.agent_config_id,
        'create_config': args.create_config,
        'config_name': args.config_name or '',
        'config_model': args.config_model or '',
        'config_description': args.config_description,
        'custom_params': '',
        'list_configs': args.list_configs,
        'list_configs_sorted': args.list_configs_sorted,
        'list_runs_sorted': args.list_runs_sorted,
        'list_summary': args.list_summary,
        'restart_searxng': args.restart_searxng,
        'containers_to_restart': containers_to_restart
    }

    # Validate create_config arguments
    if args.create_config:
        if not args.config_name or not args.config_model:
            print("Error: --config-name and --config-model are required when using --create-config")
            return 1

    run_complete_test(**kwargs)


if __name__ == "__main__":
    main()