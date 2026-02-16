"""TRM-agent pipeline alias entrypoint.

Usage:
  python -m trm_agent_pipeline.run --dataset cwq --stage preprocess
"""

from trm_agent.run import main


if __name__ == "__main__":
    main()

