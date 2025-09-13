# -*- coding: utf-8 -*-
"""Command Line Interface for UniRobot."""
from typing import List
from typing import Union

import click

from unirobot.brain import brain_launcher
from unirobot.robot import robot_launcher


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "show_default": True,
}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
def cli() -> None:
    """Command Line Interface for UniRobot."""


@cli.command()
@click.argument(
    "device-num",
    type=int,
)
@click.argument(
    "config",
    type=click.Path(exists=True),
)
@click.option(
    "-n",
    "--run-name",
    envvar="UNIROBOT_RUN_NAME",
    help="The name of the run.",
)
@click.option(
    "-it",
    "--infer-type",
    type=click.Choice(["torch", "trace", "tvm"], case_sensitive=False),
    default=None,
    help="Specify the type of infer model file, support [torch, trace, tvm].",
)
@click.option(
    "-et",
    "--export-type",
    type=click.Choice(["trace"], case_sensitive=False),
    default=None,
    help="Specify the type of export model file, support [trace] only currently.",
)
@click.option(
    "-c",
    "--ckpt",
    default=None,
    multiple=True,
    help="Specify the ckpt list for inferring.",
)
@click.option(
    "-r",
    "--resume",
    is_flag=True,
    help="Whether to continue transmission at breakpoint.",
)
@click.option(
    "-p",
    "--port",
    type=click.IntRange(1024, 65535),
    envvar="UNIROBOT_BRAIN_PORT",
    show_default="random integer",
    help="Specify the port of TCP for the distributed communication.",
)
@click.option(
    "-dm",
    "--dataset-mode",
    type=str,
    default=None,
    help="Specify the dataset mode for training or inferring.",
)
def brain_run(
    device_num: int,
    config: str,
    run_name: str,
    infer_type: str,
    export_type: str,
    ckpt: Union[None, str, List[str]],
    resume: bool,
    port: int,
    dataset_mode: str,
) -> None:
    """Entry for unirobot transmits variables from here."""
    brain_launcher.run(
        device_num,
        config,
        run_name,
        infer_type,
        export_type,
        ckpt,
        resume,
        port,
        dataset_mode,
    )


@cli.command()
@click.argument(
    "config",
    type=click.Path(exists=True),
)
@click.option(
    "-n",
    "--task-name",
    envvar="UNIROBOT_TASK_NAME",
    required=True,
    help="The name of the task.",
)
@click.option(
    "-rt",
    "--run-type",
    type=click.Choice(
        ["teleoperation", "model_local", "model_server"], case_sensitive=False
    ),
    required=True,
    help="Specify the type of run robot, support [teleoperation, model_local, model_server].",
)
@click.option(
    "-rl",
    "--use-rl",
    is_flag=True,
    help="Whether to use RL mode.",
)
def robot_run(
    config: str,
    task_name: str,
    run_type: str,
    use_rl: bool,
) -> None:
    """Entry for unirobot transmits variables from here."""
    robot_launcher.run(config, task_name, run_type, use_rl)
