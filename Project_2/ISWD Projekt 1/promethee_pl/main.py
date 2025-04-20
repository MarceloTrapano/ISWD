from pathlib import Path

import click
import numpy as np
import pandas as pd

from promethee_pl.utils import (
    load_dataset,
    load_preference_information,
    display_ranking,
)


# TODO
def calculate_marginal_preference_matrix(
    dataset: pd.DataFrame, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates the marginal preference matrix all alternatives pairs and criterion available in dataset

    :param dataset: difference between compared alternatives
    :param preference_information: preference information
    :return: 3D numpy array with marginal preference matrix on every parser, Consecutive indices [i, j, k] describe first alternative, second alternative, parser
    """
    num_alternatives = dataset.shape[0]
    num_criteria = preference_information.shape[0]
    marginal_preference = np.zeros(shape = (num_alternatives, num_alternatives, num_criteria), dtype=np.float64)
    for k in range(num_criteria):
        q, p, type = preference_information.iloc[k][["q", "p", "type"]]
        for i, j in np.ndindex(num_alternatives, num_alternatives):
            if i == j:
                marginal_preference[i, j, k] = 0
            else:
                dk = dataset.iloc[i, k] - dataset.iloc[j, k] if type == "gain" else dataset.iloc[j, k] - dataset.iloc[i, k]
                if dk >= p:
                    marginal_preference[i, j, k] = 1
                elif dk <= q:
                    marginal_preference[i, j, k] = 0
                else:
                    marginal_preference[i, j, k] = (dk - q) / (p - q)

    return marginal_preference


# TODO
def calculate_comprehensive_preference_index(
    marginal_preference_matrix: np.ndarray, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates comprehensive preference index for the given dataset

    :param marginal_preference_matrix: 3D numpy array with marginal preference matrix on every parser, Consecutive indices [i, j, k] describe first alternative, second alternative, criterion
    :param preference_information: Padnas dataframe containing preference information
    :return: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    """
    num_alternatives = marginal_preference_matrix.shape[0]
    comprehensive_preference_matrix = np.zeros(shape= (num_alternatives, num_alternatives), dtype=np.float64)
    ks = preference_information["k"].values

    for i, j in np.ndindex(num_alternatives, num_alternatives):
        if i == j:
            comprehensive_preference_matrix[i, j] = 0
        else:
            comprehensive_preference_matrix[i, j] = np.sum(ks * marginal_preference_matrix[i, j]) / np.sum(ks)

    return comprehensive_preference_matrix


# TODO
def calculate_positive_flow(
    comprehensive_preference_matrix: np.ndarray, alternatives: pd.Index
) -> pd.Series:
    """
    Function that calculates the positive flow value for the given preference matrix and corresponding index

    :param comprehensive_preference_matrix: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :return: series representing positive flow values for the given preference matrix
    """
    positive_flows = np.sum(comprehensive_preference_matrix, axis=1)
    positive_flow = pd.Series(positive_flows, index=alternatives)

    return positive_flow


# TODO
def calculate_negative_flow(
    comprehensive_preference_matrix: np.ndarray, alternatives: pd.Index
) -> pd.Series:
    """
    Function that calculates the negative flow value for the given preference matrix and corresponding index

    :param comprehensive_preference_matrix: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :return: series representing negative flow values for the given preference matrix
    """
    negative_flows = np.sum(comprehensive_preference_matrix, axis=0)
    negative_flow = pd.Series(negative_flows, index=alternatives)
    
    return negative_flow


# TODO
def calculate_net_flow(positive_flow: pd.Series, negative_flow: pd.Series) -> pd.Series:
    """
    Function that calculates the net flow value for the given positive and negative flow

    :param positive_flow: series representing positive flow values for the given preference matrix
    :param negative_flow: series representing negative flow values for the given preference matrix
    :return: series representing net flow values for the given preference matrix
    """
    net_flow = positive_flow - negative_flow

    return net_flow


# TODO
def create_partial_ranking(
    positive_flow: pd.Series, negative_flow: pd.Series
) -> pd.DataFrame:
    """
    Function that aggregates positive and negative flow to a partial ranking (from Promethee I)

    :param positive_flow: series representing positive flow values for the given preference matrix
    :param negative_flow: series representing negative flow values for the given preference matrix
    :return: partial ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    num_alternatives = positive_flow.shape[0]
    partial_ranking = np.zeros(shape=(num_alternatives, num_alternatives), dtype=np.bool)
    
    for i, j in np.ndindex(num_alternatives, num_alternatives):
        if i == j:
            partial_ranking[i, j] = True
        else:
            if positive_flow.iloc[i] >= positive_flow.iloc[j] and negative_flow.iloc[i] <= negative_flow.iloc[j]:
                partial_ranking[i, j] = True
            else:
                partial_ranking[i, j] = False
    
    partial_ranking_df = pd.DataFrame(partial_ranking, index=positive_flow.index, columns=positive_flow.index)

    return partial_ranking_df


# TODO
def create_complete_ranking(net_flow: pd.Series) -> pd.DataFrame:
    """
    Function that aggregates positive and negative flow to a complete ranking (from Promethee II)
    :param net_flow: series representing net flow values for the given preference matrix
    :return: complete ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    num_alternatives = net_flow.shape[0]
    complete_ranking = np.zeros(shape=(num_alternatives, num_alternatives), dtype=np.bool)

    for i, j in np.ndindex(num_alternatives, num_alternatives):
        if i == j:
            complete_ranking[i, j] = True
        else:
            if net_flow.iloc[i] >= net_flow.iloc[j]:
                complete_ranking[i, j] = True
            else:
                complete_ranking[i, j] = False

    complete_ranking_df = pd.DataFrame(complete_ranking, index=net_flow.index, columns=net_flow.index)

    return complete_ranking_df


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def main(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)

    dataset = load_dataset(dataset_path)
    preference_information = load_preference_information(dataset_path)

    marginal_preference_matrix = calculate_marginal_preference_matrix(
        dataset, preference_information
    )

    # compare filempm with marginal_preference_matrix
    filempm = np.load("promethee_pl\\data\\lecture\\reference\\marginal_preference_matrix.npy")
    assert np.array_equal(filempm, marginal_preference_matrix)

    comprehensive_preference_matrix = calculate_comprehensive_preference_index(
        marginal_preference_matrix, preference_information
    )

    # compare filecpm with comprehensive_preference_matrix
    filecpm = np.load("promethee_pl\\data\\lecture\\reference\\comprehensive_preference_matrix.npy")
    assert np.array_equal(filecpm, comprehensive_preference_matrix)

    positive_flow = calculate_positive_flow(
        comprehensive_preference_matrix, dataset.index
    )
    negative_flow = calculate_negative_flow(
        comprehensive_preference_matrix, dataset.index
    )

    assert positive_flow.index.equals(negative_flow.index)

    partial_ranking = create_partial_ranking(positive_flow, negative_flow)
    display_ranking(partial_ranking, "Promethee I")

    net_flow = calculate_net_flow(positive_flow, negative_flow)
    complete_ranking = create_complete_ranking(net_flow)
    display_ranking(complete_ranking, "Promethee II")


if __name__ == "__main__":
    main()
