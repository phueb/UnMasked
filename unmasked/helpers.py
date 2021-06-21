from typing import List, Tuple


blimp_file_name2phenomenon = {

    'anaphor_gender_agreement': 'anaphor agreement',
    'anaphor_number_agreement': 'anaphor agreement',

    'animate_subject_passive': 'argument structure',
    'animate_subject_trans': 'argument structure',
    'causative': 'argument structure',
    'drop_argument': 'argument structure',
    'inchoative': 'argument structure',
    'intransitive': 'argument structure',
    'passive_1': 'argument structure',
    'passive_2': 'argument structure',
    'transitive': 'argument structure',

    'principle_A_c_command': 'binding',
    'principle_A_case_1': 'binding',
    'principle_A_case_2': 'binding',
    'principle_A_domain_1': 'binding',
    'principle_A_domain_2': 'binding',
    'principle_A_domain_3': 'binding',
    'principle_A_reconstruction': 'binding',

    'existential_there_object_raising': 'control/raising',
    'existential_there_subject_raising': 'control/raising',
    'expletive_it_object_raising': 'control/raising',
    'tough_vs_raising_1': 'control/raising',
    'tough_vs_raising_2': 'control/raising',

    'determiner_noun_agreement_1': 'determiner-noun agreement',
    'determiner_noun_agreement_2': 'determiner-noun agreement',
    'determiner_noun_agreement_irregular_1': 'determiner-noun agreement',
    'determiner_noun_agreement_irregular_2': 'determiner-noun agreement',
    'determiner_noun_agreement_with_adjective_1': 'determiner-noun agreement',
    'determiner_noun_agreement_with_adj_2': 'determiner-noun agreement',
    'determiner_noun_agreement_with_adj_irregular_1': 'determiner-noun agreement',
    'determiner_noun_agreement_with_adj_irregular_2': 'determiner-noun agreement',

    'ellipsis_n_bar_1': 'ellipsis',
    'ellipsis_n_bar_2': 'ellipsis',

    'wh_questions_object_gap': 'filler-gap',
    'wh_questions_subject_gap': 'filler-gap',
    'wh_questions_subject_gap_long_distance': 'filler-gap',
    'wh_vs_that_no_gap': 'filler-gap',
    'wh_vs_that_no_gap_long_distance': 'filler-gap',
    'wh_vs_that_with_gap': 'filler-gap',
    'wh_vs_that_with_gap_long_distance': 'filler-gap',

    'irregular_past_participle_adjectives': 'irregular forms',
    'irregular_past_participle_verbs': 'irregular forms',

    'distractor_agreement_relational_noun': 'subject-verb agreement',
    'distractor_agreement_relative_clause': 'subject-verb agreement',
    'irregular_plural_subject_verb_agreement_1': 'subject-verb agreement',
    'irregular_plural_subject_verb_agreement_2': 'subject-verb agreement',
    'regular_plural_subject_verb_agreement_1': 'subject-verb agreement',
    'regular_plural_subject_verb_agreement_2': 'subject-verb agreement',

    'existential_there_quantifiers_1': 'quantifiers',
    'existential_there_quantifiers_2': 'quantifiers',
    'superlative_quantifiers_1': 'quantifiers',
    'superlative_quantifiers_2': 'quantifiers',

    'matrix_question_npi_licensor_present': 'npi licensing',
    'npi_present_1': 'npi licensing',
    'npi_present_2': 'npi licensing',
    'only_npi_licensor_present': 'npi licensing',
    'only_npi_scope': 'npi licensing',
    'sentential_negation_npi_licensor_present': 'npi licensing',
    'sentential_negation_npi_scope': 'npi licensing',

    'adjunct_island': 'island effects',
    'complex_NP_island': 'island effects',
    'coordinate_structure_constraint_complex_left_branch': 'island effects',
    'coordinate_structure_constraint_object_extraction': 'island effects',
    'left_branch_island_echo_question': 'island effects',
    'left_branch_island_simple_question': 'island effects',
    'sentential_subject_island': 'island effects',
    'wh_island': 'island effects',

}

zorro_file_name2phenomenon = {

    'anaphor_agreement-pronoun_gender': 'anaphor agreement',

    'argument_structure-dropped_argument': 'argument structure',
    'argument_structure-swapped_arguments': 'argument structure',
    'argument_structure-transitive': 'argument structure',

    'binding-principle_a': 'binding',

    'case-subjective_pronoun': 'case',  # not in BLiMP

    'local_attractor-in_question_with_aux': 'local_attractor',  # not in BLiMP

    'agreement_determiner_noun-across_1_adjective': 'determiner-noun agreement',
    'agreement_determiner_noun-between_neighbors': 'determiner-noun agreement',

    'ellipsis-n_bar': 'ellipsis',

    'filler-gap-wh_question_object': 'filler-gap',
    'filler-gap-wh_question_subject': 'filler-gap',

    'irregular-verb': 'irregular',

    'agreement_subject_verb-across_prepositional_phrase': 'subject-verb agreement',
    'agreement_subject_verb-across_relative_clause': 'subject-verb agreement',
    'agreement_subject_verb-in_question_with_aux': 'subject-verb agreement',
    'agreement_subject_verb-in_simple_question': 'subject-verb agreement',

    'quantifiers-existential_there': 'quantifiers',
    'quantifiers-superlative': 'quantifiers',

    'npi_licensing-matrix_question': 'npi licensing',
    'npi_licensing-only_npi_licensor': 'npi licensing',

    'island-effects-adjunct_island': 'island effects',
    'island-effects-coordinate_structure_constraint': 'island effects',

}

zorro_phenomena = sorted([v for v in zorro_file_name2phenomenon.values()])
blimp_phenomena = sorted([v for v in zorro_file_name2phenomenon.values()])
zorro_phenomenon2fns = {phenomenon: [pa for pa, ph in zorro_file_name2phenomenon.items() if ph == phenomenon]
                        for phenomenon in zorro_phenomena}
blimp_phenomenon2fns = {phenomenon: [pa for pa, ph in blimp_file_name2phenomenon.items() if ph == phenomenon]
                        for phenomenon in blimp_phenomena}


def shorten_phenomenon(phenomenon: str,
                       ) -> str:
    phenomenon = phenomenon.replace('determiner-noun', 'det. noun')
    phenomenon = phenomenon.replace('agreement', 'agr.')
    phenomenon = phenomenon.replace('argument', 'arg.')
    phenomenon = phenomenon.replace('coordinate', 'coord.')
    phenomenon = phenomenon.replace('structure', 'struct.')
    phenomenon = phenomenon.replace('attractor', 'attr.')
    phenomenon = phenomenon.replace('licensing', 'lic.')

    return phenomenon
