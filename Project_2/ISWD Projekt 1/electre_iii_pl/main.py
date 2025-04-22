from pathlib import Path

import click
import numpy as np
import pandas as pd

from electre_iii_pl.utils import (
    load_dataset,
    load_preference_information,
    display_ranking,
)


# TODO
def calculate_marginal_concordance_matrix(
    dataset: pd.DataFrame, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates the marginal concordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param preference_information: pandas dataframe with preference information
    :return: 3D numpy array with marginal concordance matrix with shape [number of alternatives, number of alternatives, number of criterion], where element with index [i, j, k] describe marginal concordance index between alternative i and alternative j on criterion k
    """
    num_alternatives = dataset.shape[0]
    num_criteria = preference_information.shape[0]
    marginal_concordance_matrix = np.zeros(shape = (num_alternatives, num_alternatives, num_criteria), dtype=np.float64)

    for k in range(num_criteria):
        q, p, type = preference_information.iloc[k][["q", "p", "type"]]

        for i, j in np.ndindex(num_alternatives, num_alternatives):
            if i == j:
                marginal_concordance_matrix[i, j, k] = 1
            else:
                dk = dataset.iloc[i, k] - dataset.iloc[j, k] if type == "gain" else dataset.iloc[j, k] - dataset.iloc[i, k]
                if dk >= -q:
                    marginal_concordance_matrix[i, j, k] = 1
                elif dk <= -p:
                    marginal_concordance_matrix[i, j, k] = 0
                else:
                    marginal_concordance_matrix[i, j, k] = (dk + p) / (p - q)

    return marginal_concordance_matrix


# TODO
def calculate_comprehensive_concordance_matrix(
    marginal_concordance_matrix: np.ndarray, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates comprehensive concordance matrix for the given dataset

    :param marginal_concordance_matrix: 3D numpy array with marginal concordance matrix with shape [number of alternatives, number of alternatives, number of criterion], where element with index [i, j, k] describe marginal concordance index between alternative i and alternative j on criterion k
    :param preference_information: pandas dataframe with preference information
    :return: 2D numpy array with comprehensive concordance matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe comprehensive concordance index between alternative i and alternative j
    """
    num_alternatives = marginal_concordance_matrix.shape[0]
    comprehensive_concordance_matrix = np.zeros(shape= (num_alternatives, num_alternatives), dtype=np.float64)
    ks = preference_information["k"].values

    for i, j in np.ndindex(num_alternatives, num_alternatives):
        if i == j:
            comprehensive_concordance_matrix[i, j] = 1
        else:
            comprehensive_concordance_matrix[i, j] = np.sum(ks * marginal_concordance_matrix[i, j]) / np.sum(ks)

    return comprehensive_concordance_matrix


# TODO
def calculate_marginal_discordance_matrix(
    dataset: pd.DataFrame, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates the marginal discordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param preference_information: pandas dataframe with preference information
    :return: 3D numpy array with marginal discordance matrix with shape [number of alternatives, number of alternatives, number of criterion], where element with index [i, j, k] describe marginal discordance index between alternative i and alternative j on criterion k
    """
    num_alternatives = dataset.shape[0]
    num_criteria = preference_information.shape[0]
    marginal_discordance_matrix = np.zeros(shape = (num_alternatives, num_alternatives, num_criteria), dtype=np.float64)

    for k in range(num_criteria):
        v, p, type = preference_information.iloc[k][["v", "p", "type"]]

        for i, j in np.ndindex(num_alternatives, num_alternatives):
            if i == j:
                marginal_discordance_matrix[i, j, k] = 0
            else:
                dk = dataset.iloc[i, k] - dataset.iloc[j, k] if type == "gain" else dataset.iloc[j, k] - dataset.iloc[i, k]
                if dk <= -v:
                    marginal_discordance_matrix[i, j, k] = 1
                elif dk >= -p:
                    marginal_discordance_matrix[i, j, k] = 0
                else:
                    marginal_discordance_matrix[i, j, k] = -(dk + p) / (v - p)

    return marginal_discordance_matrix


# TODO
def calculate_credibility_index(
    comprehensive_concordance_matrix: np.ndarray,
    marginal_discordance_matrix: np.ndarray,
) -> np.ndarray:
    """
    Function that calculates the credibility index for the given comprehensive concordance matrix and marginal discordance matrix

    :param comprehensive_concordance_matrix: 2D numpy array with comprehensive concordance matrix. Every entry in the matrix [i, j] represents comprehensive concordance index between alternative i and alternative j
    :param marginal_discordance_matrix: 2D numpy array with marginal discordance matrix, Consecutive indices [i, j, k] describe first alternative, second alternative, criterion
    :return: 2D numpy array with credibility matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe credibility index between alternative i and alternative j
    """
    num_alternatives = comprehensive_concordance_matrix.shape[0]
    credibility_index = np.zeros(shape= (num_alternatives, num_alternatives), dtype=np.float64)

    for i, j in np.ndindex(num_alternatives, num_alternatives):
        if i == j:
            credibility_index[i, j] = 0
        else:
            d_above_c_index = np.where(marginal_discordance_matrix[i, j] > comprehensive_concordance_matrix[i, j])[0]
            marginal_discordances_above_concordance = marginal_discordance_matrix[i, j][d_above_c_index]
            cc_score = comprehensive_concordance_matrix[i, j]

            credibility_index[i, j] = cc_score * np.prod((1 - marginal_discordances_above_concordance) / (1 - cc_score))
    
    return credibility_index


# TODO
def descending_distillation(
    credibility_index: np.ndarray,
    alternatives: pd.Index,
    alpha: float = -0.15,
    beta: float = 0.3,
) -> pd.DataFrame:
    """
    Function that calculates the descending distillation procedure

    :param credibility_index: 2D numpy array with credibility matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe credibility index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :param alpha: the parameter alpha for the s function
    :param beta: the parameter beta for the s function
    :return: descending ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    lambda_k=np.max(credibility_index)
    descending_ranking = descending_distillation_start_lambda(lambda_k=lambda_k, credibility_index=credibility_index, alternatives=alternatives, alpha=alpha, beta=beta)
    descending_ranking = pd.DataFrame(descending_ranking, index=alternatives, columns=alternatives)

    return descending_ranking

def descending_distillation_start_lambda(
    lambda_k: float,
    credibility_index: np.ndarray,
    alternatives: pd.Index,
    alpha: float = -0.15,
    beta: float = 0.3,
    internal: bool = False,
) -> pd.DataFrame:
    """
    Function that calculates the descending distillation procedure

    :param lambda_k: the initial value of lambda_k
    :param credibility_index: 2D numpy array with credibility matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe credibility index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :param alpha: the parameter alpha for the s function
    :param beta: the parameter beta for the s function
    :return: descending ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    num_alternatives = credibility_index.shape[0]
    descending_ranking = np.eye(num_alternatives, num_alternatives, dtype=np.bool)
    unranked_alternatives = [i for i in range(alternatives.shape[0])]
    credibility_index_unranked = credibility_index[np.ix_(unranked_alternatives, unranked_alternatives)]
    
    while len(unranked_alternatives) > 0:
        reversed_credibility_index_unranked = credibility_index_unranked.T
        s_ab = credibility_index_unranked * alpha + beta # wartość s dla każdej pary
        s_k = alpha * lambda_k + beta

        if lambda_k == 0:
            # finish distillation - no best alternative
            descending_ranking[np.ix_(unranked_alternatives, unranked_alternatives)] = True
            unranked_alternatives = []
            continue

        lambda_k = np.max(credibility_index_unranked[credibility_index_unranked < lambda_k - s_k])

        considered_relations = np.where((credibility_index_unranked > lambda_k) & (credibility_index_unranked > reversed_credibility_index_unranked + s_ab))

        strength = np.asarray([np.sum(considered_relations[0] == i) for i in range(len(unranked_alternatives))], dtype=np.int64)
        weakness = np.asarray([np.sum(considered_relations[1] == i) for i in range(len(unranked_alternatives))], dtype=np.int64)
        quality = strength - weakness

        best_alternatives_index = np.asarray(unranked_alternatives)[np.where(np.max(quality) == quality)[0]]

        if len(best_alternatives_index) > 1:
            # internal distillation
            internal_cred_index = credibility_index[np.ix_(best_alternatives_index, best_alternatives_index)]
            internal_alternatives = alternatives[best_alternatives_index]

            internal_dist_res = descending_distillation_start_lambda(lambda_k=lambda_k, credibility_index=internal_cred_index, alternatives=internal_alternatives, alpha=alpha, beta=beta, internal=True)

            internal_best_alternatives_index = best_alternatives_index[np.sum(internal_dist_res, axis=1) == len(internal_alternatives)]
            best_alternatives_index = internal_best_alternatives_index

        # update ranking
        descending_ranking[np.ix_(best_alternatives_index, unranked_alternatives)] = True
        for alt in best_alternatives_index:
            unranked_alternatives.remove(alt)
        
        if len(unranked_alternatives) == 0:
            continue
        # update variables
        credibility_index_unranked = credibility_index[np.ix_(unranked_alternatives, unranked_alternatives)]
        lambda_k = np.max(credibility_index_unranked)

        # internal distillation finishes after one iteration
        if internal:
            break

    return descending_ranking


# TODO
def ascending_distillation(
    credibility_index: np.ndarray,
    alternatives: pd.Index,
    alpha: float = -0.15,
    beta: float = 0.3,
) -> pd.DataFrame:
    """
    Function that calculates the ascending distillation procedure

    :param credibility_index: 2D numpy array with credibility matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe credibility index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :param alpha: the parameter alpha for the s function
    :param beta: the parameter beta for the s function
    :return: ascending ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    lambda_k=np.max(credibility_index)
    ascending_ranking = ascending_distillation_start_lambda(lambda_k=lambda_k, credibility_index=credibility_index, alternatives=alternatives, alpha=alpha, beta=beta)
    ascending_ranking = pd.DataFrame(ascending_ranking, index=alternatives, columns=alternatives)

    return ascending_ranking

def ascending_distillation_start_lambda(
    lambda_k: float,
    credibility_index: np.ndarray,
    alternatives: pd.Index,
    alpha: float = -0.15,
    beta: float = 0.3,
    internal: bool = False,
) -> pd.DataFrame:
    """
    Function that calculates the ascending distillation procedure

    :param lambda_k: the initial value of lambda_k
    :param credibility_index: 2D numpy array with credibility matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe credibility index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :param alpha: the parameter alpha for the s function
    :param beta: the parameter beta for the s function
    :return: ascending ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    num_alternatives = credibility_index.shape[0]
    ascending_ranking = np.eye(num_alternatives, num_alternatives, dtype=np.bool)
    unranked_alternatives = [i for i in range(alternatives.shape[0])]
    credibility_index_unranked = credibility_index[np.ix_(unranked_alternatives, unranked_alternatives)]
    
    while len(unranked_alternatives) > 0:
        reversed_credibility_index_unranked = credibility_index_unranked.T
        s_ab = credibility_index_unranked * alpha + beta # wartość s dla każdej pary
        s_k = alpha * lambda_k + beta

        if lambda_k == 0:
            # finish distillation - no best alternative
            ascending_ranking[np.ix_(unranked_alternatives, unranked_alternatives)] = True
            unranked_alternatives = []
            continue

        lambda_k = np.max(credibility_index_unranked[credibility_index_unranked < lambda_k - s_k])

        considered_relations = np.where((credibility_index_unranked > lambda_k) & (credibility_index_unranked > reversed_credibility_index_unranked + s_ab))

        strength = np.asarray([np.sum(considered_relations[0] == i) for i in range(len(unranked_alternatives))], dtype=np.int64)
        weakness = np.asarray([np.sum(considered_relations[1] == i) for i in range(len(unranked_alternatives))], dtype=np.int64)
        quality = strength - weakness

        best_alternatives_index = np.asarray(unranked_alternatives)[np.where(np.min(quality) == quality)[0]]

        if len(best_alternatives_index) > 1:
            # internal distillation
            internal_cred_index = credibility_index[np.ix_(best_alternatives_index, best_alternatives_index)]
            internal_alternatives = alternatives[best_alternatives_index]

            internal_dist_res = ascending_distillation_start_lambda(lambda_k=lambda_k, credibility_index=internal_cred_index, alternatives=internal_alternatives, alpha=alpha, beta=beta, internal=True)

            internal_best_alternatives_index = best_alternatives_index[np.sum(internal_dist_res, axis=0) == len(internal_alternatives)]
            best_alternatives_index = internal_best_alternatives_index

        # update ranking
        ascending_ranking[np.ix_(unranked_alternatives, best_alternatives_index)] = True
        for alt in best_alternatives_index:
            unranked_alternatives.remove(alt)

        if len(unranked_alternatives) == 0:
            continue
        # update variables
        credibility_index_unranked = credibility_index[np.ix_(unranked_alternatives, unranked_alternatives)]
        lambda_k = np.max(credibility_index_unranked)

        # internal distillation finishes after one iteration
        if internal:
            break

    return ascending_ranking

# TODO
def create_final_ranking(
    descending_ranking: pd.DataFrame, ascending_ranking: pd.DataFrame
) -> pd.DataFrame:
    """
    Function that computes the final ranking from both ascending and descending ranking

    :param descending_ranking: dataframe representing descending ranking
    :param ascending_ranking: dataframe representing ascending ranking
    :return: final ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    final_ranking = descending_ranking & ascending_ranking

    return final_ranking


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def main(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)

    dataset = load_dataset(dataset_path)
    preference_information = load_preference_information(dataset_path)

    marginal_concordance_matrix = calculate_marginal_concordance_matrix(
        dataset, preference_information
    )
    comprehensive_concordance_matrix = calculate_comprehensive_concordance_matrix(
        marginal_concordance_matrix, preference_information
    )

    marginal_discordance_matrix = calculate_marginal_discordance_matrix(
        dataset, preference_information
    )
    credibility_index = calculate_credibility_index(
        comprehensive_concordance_matrix, marginal_discordance_matrix
    )

    descending_ranking = descending_distillation(credibility_index, dataset.index)
    display_ranking(descending_ranking, "Descending Ranking") # -> problem z '\n' w nazwach nodów, cokolwiek innego działa; może u mnie wersja pygraphviz jest inna czy coś

    ascending_ranking = ascending_distillation(credibility_index, dataset.index)
    display_ranking(ascending_ranking, "Ascending Ranking")

    final_ranking = create_final_ranking(descending_ranking, ascending_ranking)
    display_ranking(final_ranking, "Final Ranking")


if __name__ == "__main__":
    main()
