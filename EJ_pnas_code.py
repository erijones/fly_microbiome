#!/usr/bin/env python3
#
# EJ_pnas_code.py :: Version 1.0
#
# Here we include scripts that were used to create figures for the paper
# "Microbiome interactions shape host fitness", by Gould et al., submitted to
# PNAS. We provide scripts for Figures 2 and 6 of the main text, as well as
# supplemental figures S15ABDE and S17
#
# Send questions to Eric Jones at ewj@physics.ucsb.edu
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import itertools





def make_fig_2():
    """ This function creates Figure 2 of the main text, which creates a
    "mixing model" that predicts host traits of a given microbial composition
    by averaging the traits of a single microbial species """
    # load fly phenotype summary data
    with open('SummaryDataTable_052018.csv','r') as f:
        trait_data = [line.strip().split(",") for line in f]
    toc = trait_data[0] # "table of contents" toc
    trait_data = np.array(trait_data)
    # binary ID gives presence/absence of each of 5 microbial species
    binary_ids = trait_data[1:, toc.index("Binary ID")]
    # compute diversity (=N) of each binary ID
    Ns = np.array([sum([int(x) for x in bin_id]) for bin_id in binary_ids])

    # get traits as functions of microbiome compositions
    fecundity_mean = np.array([float(x) for x in trait_data[1:, toc.index("Daily Fecundity (SE)")-1]])
    death_mean = np.array([float(x) for x in trait_data[1:, toc.index("Time to Death (SE)")-1]])
    development_mean = np.array([float(x) for x in trait_data[1:, toc.index("Development Time (SE)")-1]])
    bacterial_mean = np.array([float(x) for x in trait_data[1:, toc.index("Bacterial Load (SE)")-2]])

    # get trait errors as functions of microbiome compositions
    fecundity_se = np.array([float(x) for x in trait_data[1:, toc.index("Daily Fecundity (SE)")]])
    death_se = np.array([float(x) for x in trait_data[1:, toc.index("Time to Death (SE)")]])
    development_se = np.array([float(x) for x in trait_data[1:, toc.index("Development Time (SE)")]])
    bacterial_se = np.array([float(x) for x in trait_data[1:, toc.index("Bacterial Load (SE)")]])

    ### PART 1: MIXING MODEL PREDICTIONS FOR 1-SPECIES AND 2-SPECIES INTERACTIONS 
    print('CREATING PLOT FOR NAIVE 1-SPECIES AND 2-SPECIES MIXING MODELS')
    for j,(trait, err, label) in enumerate(
            zip([fecundity_mean, death_mean, development_mean, bacterial_mean],
                [fecundity_se, death_se, development_se, bacterial_se],
                ['daily fecundity (eggs)', 'time to death (days)',
                    'development time (days)', 'bacterial load (CFUs)'])):
        ax = plt.subplot(2, 2, j+1)

        # add offset to each diversity N (so that all traits are collapsed)
        offsets = []
        for i in range(0, 6):
            num_combs = scipy.special.binom(5, i)
            if num_combs == 1:
                offsets.append(0)
                continue
            sub_offsets = np.linspace(-.4, .4, int(num_combs))
            offsets.extend(sub_offsets)
        offsets = np.array(offsets)
        # offset_Ns is the "x-axis" data for each trait
        offset_Ns = offsets + Ns

        # plot naive non-interaction model predictions
        # this model assumes trait(11000) = 1/2*(trait(10000) + trait(01000))
        single_traits = trait[1:6]
        single_trait_errors = err[1:6]
        noninteracting_predictions = []
        noninteracting_err_predictions = []
        for k,bin_id in enumerate(binary_ids[1:]):
            # trait_val is the single-species mixing prediction for each binary ID
            trait_val = 0
            variance = 0
            N = sum([int(x) for x in bin_id])
            if N < 2:
                # only predict compositions of N >= 2
                continue
            for i,elem in enumerate(bin_id):
                if int(elem):
                    trait_val += single_traits[i]/N
                    variance += single_trait_errors[i]**2/N
            noninteracting_predictions.append(trait_val)
            noninteracting_err_predictions.append(np.sqrt(variance))

        # plot mixing model pairwise interactions predictions
        # this model assumes trait(11100)=1/3*(trait(11000)+trait(10100)+trait(01100))
        bi_traits = {}
        bi_errors = {}
        for i,bin_id in enumerate(binary_ids[1:]):
            N = sum([int(x) for x in bin_id])
            if N == 2:
                indices = tuple([k for k,x in enumerate(bin_id) if int(x) == 1])
                bi_traits[indices] = trait[i+1]
                bi_errors[indices] = err[i+1]

        pairwise_traits = []
        pairwise_errors = []
        for bin_id in binary_ids[1:]:
            N = sum([int(x) for x in bin_id])
            if N < 3:
                # only predict compositions of N >= 3
                continue
            indices = [i for i,x in enumerate(bin_id) if int(x) == 1]
            inner_pairs = list(itertools.combinations(indices, 2))
            num_pairs = len(inner_pairs)
            pairwise_trait = 0
            pairwise_error = 0
            for pair in inner_pairs:
                pairwise_trait += 1/num_pairs * bi_traits[pair]
                pairwise_error += 1/num_pairs * bi_errors[pair]**2
            pairwise_traits.append(pairwise_trait)
            pairwise_errors.append(np.sqrt(pairwise_error))

        # plot experimental errors
        ax1 = ax.errorbar(offset_Ns[1:], trait[1:],
                          yerr=[1.96*x for x in err[1:]], fmt='.',
                          capsize=2, lw=1, ms=8, label='measured', zorder=4)

        ax2 = ax.errorbar(offset_Ns[6:], noninteracting_predictions,
                          yerr=[1.96*x for x in noninteracting_err_predictions],
                          fmt='D', capsize=2, lw=.4, ms=4, mfc='white',
                          mew=.5, label='predicted (single species)',
                          zorder=3)

        ax3 = ax.errorbar(offset_Ns[16:], pairwise_traits,
                          yerr=[1.96*x for x in pairwise_errors], fmt='x',
                          capsize=2, lw=.4, ms=4, mfc='white', mew=.5,
                          label='predicted (pairwise)', zorder=3)

        # plot separator bars for different diversities
        for x in [1.5, 2.5, 3.5, 4.5]:
            ax.axvline(x, color='k', ls='--', lw=.5)

        # format plots
        ax.set_xticklabels([1, 2, 3, 4, 5], fontsize=12)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(.4, 5.4)
        ax.set_ylabel(label, fontsize=12)
        if j == 2 or j == 3:
            ax.set_xlabel('N', fontsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),
            fontsize=12)
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.legend(fontsize=10, ncol=3,
            loc='upper right', bbox_to_anchor = (.85, 2.65))
    plt.subplots_adjust(left=.1, right=.97, hspace=.3, wspace=.3)
    plt.savefig('figs/Figure_2_predictions.pdf')

    ### PART 2: DIFFERENCE BETWEEN MIXING MODEL AND EXPERIMENTAL DATA
    # largely identical to PART 1, but I keep track of differences between the
    # model predictions and experimental measurements
    plt.figure()
    print('CREATING PLOT FOR NAIVE 1-SPECIES AND 2-SPECIES MIXING MODELS DIFFERENCES')
    single_total = 0
    single_captured = 0
    pairwise_total = 0
    pairwise_captured = 0
    for j,(trait, err, label) in enumerate(
            zip([fecundity_mean, death_mean, development_mean, bacterial_mean],
                [fecundity_se, death_se, development_se, bacterial_se],
                ['daily fecundity (eggs)', 'time to death (days)',
                    'development time (days)', 'bacterial load (CFUs)'])):
        ax = plt.subplot(2, 2, j+1)

        # add offset to each diversity N (so that all traits are collapsed)
        offsets = []
        for i in range(0, 6):
            num_combs = scipy.special.binom(5, i)
            if num_combs == 1:
                offsets.append(0)
                continue
            sub_offsets = np.linspace(-.4, .4, int(num_combs))
            offsets.extend(sub_offsets)
        offsets = np.array(offsets)
        offset_Ns = offsets + Ns

        # plot naive non-interaction model predictions
        single_traits = trait[1:6]
        single_trait_errors = err[1:6]
        noninteracting_predictions = []
        noninteracting_err_predictions = []
        for k,bin_id in enumerate(binary_ids[1:]):
            trait_val = 0
            variance = 0
            N = sum([int(x) for x in bin_id])
            if N < 2:
                continue
            for i,elem in enumerate(bin_id):
                if int(elem):
                    trait_val += single_traits[i]/N
                    # variance of model prediction:
                    variance += single_trait_errors[i]**2/N
            # variance of experimental measurement:
            variance += err[k+1]**2
            noninteracting_predictions.append(trait_val - trait[k+1])
            noninteracting_err_predictions.append(np.sqrt(variance))

        bi_traits = {}
        bi_errors = {}
        for i,bin_id in enumerate(binary_ids[1:]):
            N = sum([int(x) for x in bin_id])
            if N == 2:
                indices = tuple([k for k,x in enumerate(bin_id) if int(x) == 1])
                bi_traits[indices] = trait[i+1]
                bi_errors[indices] = err[i+1]

        pairwise_traits = []
        pairwise_errors = []
        for k,bin_id in enumerate(binary_ids[1:]):
            N = sum([int(x) for x in bin_id])
            if N < 3:
                continue
            indices = [i for i,x in enumerate(bin_id) if int(x) == 1]
            inner_pairs = list(itertools.combinations(indices, 2))
            num_pairs = len(inner_pairs)
            pairwise_trait = 0
            pairwise_error = 0
            for pair in inner_pairs:
                pairwise_trait += 1/num_pairs * bi_traits[pair]
                # variance of model prediction:
                pairwise_error += 1/num_pairs * bi_errors[pair]**2
            # variance of experimental measurement:
            pairwise_error += err[k+1]**2
            pairwise_traits.append(pairwise_trait - trait[k+1])
            pairwise_errors.append(np.sqrt(pairwise_error))

        single_traits = noninteracting_predictions[10:]
        # confidence intervals 'cis'
        single_trait_cis =  [1.96*x for x in noninteracting_err_predictions[10:]]
        pairwise_trait_cis = [1.96*x for x in pairwise_errors]

        # skip first color (for presentation purposes):
        next(ax._get_lines.prop_cycler)
        ax2 = ax.errorbar(offset_Ns[6:], noninteracting_predictions,
                          yerr= [1.96*x for x in noninteracting_err_predictions], fmt='D',
                          capsize=2, lw=.4, ms=4, mfc='white', mew=.5,
                          label='predicted - measured (single species)')

        ax3 = ax.errorbar(offset_Ns[16:], pairwise_traits,
                          yerr=[1.96*x for x in pairwise_errors], fmt='x',
                          capsize=2, lw=.4, ms=4, mfc='white', mew=.5,
                          label='predicted - measured (pairwise)')

        # format plot 
        for x in [2.5, 3.5, 4.5]:
            ax.axvline(x, color='k', ls='--', lw=.5)

        ax.set_xticklabels([2, 3, 4, 5], fontsize=12)
        ax.set_xticks([2, 3, 4, 5])
        ax.set_xlim(1.4, 5.4)
        ax.set_ylabel(label, fontsize=12)
        if j == 2 or j == 3:
            ax.set_xlabel('N', fontsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.axhline(y=0, color='k', lw=.5)

        ### PART 3: COMPUTE STATISTICS
        # computes how many of the experimental data points lie within the
        # confidence intervals of the model predictions

        single_subcapture = 0 # how many measurements are captured with the 1-species model
        single_subtotal = 0 # how many measurements there are total
        pairwise_subcapture = 0 # how many measurements are captured with the 2-species model
        pairwise_subtotal = 0 # how many measurements there are total
        for trait,ci in zip(single_traits, single_trait_cis):
            single_subtotal += 1
            if (trait - ci) < 0 and (trait + ci) > 0:
                # if the model captures the data point:
                single_subcapture += 1
        for trait,ci in zip(pairwise_traits, pairwise_trait_cis):
            pairwise_subtotal += 1
            if (trait - ci) < 0 and (trait + ci) > 0:
                # if the model captures the data point:
                pairwise_subcapture += 1
        single_captured += single_subcapture
        single_total += single_subtotal
        pairwise_captured += pairwise_subcapture
        pairwise_total += pairwise_subtotal

        # compute and present statistics
        print(label)
        print('    single-species prediction captured {} out of {} data points (95% CI)'.
              format(single_subcapture, single_subtotal))
        print('    pairwise prediction captured {} out of {} data points (95% CI)'.
              format(pairwise_subcapture, pairwise_subtotal))
        print('    Fisher\'s exact test: p={}'.format(scipy.stats.fisher_exact(
              [[single_subcapture, pairwise_subcapture],
               [single_subtotal - single_subcapture,
                pairwise_subtotal - pairwise_subcapture]])[1]))
        print()
        print('    1-species mean error: {}, std dev: {}'.format(
            np.mean([abs(x) for x in noninteracting_predictions[10:]]),
            np.std([abs(x) for x in noninteracting_predictions[10:]])))
        print('    2-species mean error: {}, std dev: {}'.format(
              np.mean([abs(x) for x in pairwise_traits]),
              np.std([abs(x) for x in pairwise_errors])))
        t_test_val = scipy.stats.ttest_ind([abs(x) for x in noninteracting_predictions[10:]],
                                           [abs(x) for x in pairwise_traits], equal_var=False)
        print('    Welch\'s t={}, p={}'.format(t_test_val[0], t_test_val[1]))

    # format plot
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),
            fontsize=12)
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.legend(fontsize=10, ncol=3,
            loc='upper right', bbox_to_anchor = (.87, 2.65))
    plt.subplots_adjust(left=.12, right=.97, hspace=.3, wspace=.3)
    plt.savefig('figs/Figure_2_prediction_errors.pdf')

    # summarize statistics (over all traits)
    print('FOR ALL TRAITS:')
    print('    single-species prediction captured {} out of {} data points'.
          format(single_captured, single_total))
    print('    pairwise prediction captured {} out of {} data points'.
          format(pairwise_captured, pairwise_total))
    print('    Fisher\'s exact test: p={}'.format(scipy.stats.fisher_exact(
          [[single_captured, pairwise_captured],
           [single_total - single_captured,
            pairwise_total - pairwise_captured]])[1]))


if __name__ == '__main__':
    make_fig_2()

