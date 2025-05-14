
import os
import enum
import pandas as pd
import json

class InteractionType(enum.Enum):
    execution = 0
    submission = enum.auto()

class DatasetType(enum.Enum):
    problem = 0
    testcase = enum.auto()
    execution_interaction = enum.auto()
    submission_interaction = enum.auto()
    execution_code = enum.auto()
    submission_code = enum.auto()
    ai_error_feedback = enum.auto()
    problem_attempt_sequence = enum.auto()
    attempt_interaction_sequence_pkl = enum.auto()

final_dataset_group_fnames = {
        DatasetType.problem: 'problem.csv',
        DatasetType.testcase: 'testcase.csv',
        DatasetType.execution_interaction: 'execution_interaction.csv',
        DatasetType.submission_interaction: 'submission_interaction.csv',
        DatasetType.execution_code: 'execution_code.csv',
        DatasetType.submission_code: 'submission_code.csv',
        DatasetType.ai_error_feedback: 'ai_error_feedback.csv',
        DatasetType.problem_attempt_sequence: 'problem_attempt_sequence.csv',
        DatasetType.attempt_interaction_sequence_pkl: 'attempt_interaction_sequence.pkl'
    }

def get_dataset_path(dataset_root_dir:str, dataset: DatasetType):
    fname = final_dataset_group_fnames[dataset]
    return os.path.join(dataset_root_dir, fname)

def get_semester_groups(dataset_root_dir:str):
    csv_path = get_dataset_path(dataset_root_dir, DatasetType.problem_attempt_sequence)
    df = pd.read_csv(csv_path)
    return sorted(list(df["semester_group"].unique())), df

def get_semester_group_df(df:pd.DataFrame, semester_group:str):
    return df[df["semester_group"] == semester_group]
