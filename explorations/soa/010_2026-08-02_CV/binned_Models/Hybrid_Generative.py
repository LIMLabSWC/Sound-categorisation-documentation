import numpy as np
from scipy.stats import norm
from scipy.integrate import trapezoid
from BE import Noise_generator,Delta_repulsion,find_closest_element, calculate_cdf


def hybrid_calc(x, s_hat, categories, gamma, sigma_update, y_BE, Delta_learning, eta_learning, alpha,lambda_A, lambda_B, no_response, seed):

    # 1. Initialize INDEPENDENT random number generators
    # We add a unique offset to the seed for each generator so they are distinct but reproducible
    rng_Final = np.random.default_rng(2 * seed)
    rng_SC = np.random.default_rng(2 * seed )
    rng_BE = np.random.default_rng(2 * seed )


    n = len(s_hat) # Number of trials

    # Initialize the probability distributions for Category A and Category B.
    # A_initial_distribution: A Gaussian distribution centered on the lower half of the stimulus space.
    A_initial_distribution = norm.pdf(x, loc=-0.75, scale=0.5)
    # B_initial_distribution: A Gaussian distribution centered on the upper half of the stimulus space.
    B_initial_distribution = norm.pdf(x, loc=0.75, scale=0.5)

    # Copy the initial distributions to use for updates.
    A_distribution = A_initial_distribution.copy()
    B_distribution = B_initial_distribution.copy()

    # Initialize a list to store the subject's responses (choices).
    Response_BE = []    # Store subject's choices (0 for A, 1 for B) --BE
    ProbB_BE = []     # Store the probability of choosing category B --BE
    Response_SC = []    # Store subject's choices (0 for A, 1 for B) --SC
    ProbB_SC = []     # Store the probability of choosing category B --SC

    Final_Response = []
    Final_ProbB = []




    # Iterate through each trial to compute choice and update belief distributions.
    for i in range(n):
        # Proceed only if the subject responded in the trial (no_response is False).
        if no_response[i] == False:
            # Get the perceived stimulus for the current trial and find the closest element in x.
            s = s_hat[i].astype(np.float32)
            j = find_closest_element(x, s)


            ####################################################################################################
            ######                                       SC part                                          ######
            ####################################################################################################

            # Calculate the probability of choosing Category A (P_A) and Category B (P_B).
            # Safe division guard: if the sum of distributions is zero, set P_A and P_B to 0.5.
            denom = A_distribution[j] + B_distribution[j]
            if denom <= 0:
                print(f"[calc_choice] WARNING trial={i}, stim={s:.4f}, bin_idx={j}, "
                      f"x[{j}]={x[j]:.4f}, A={A_distribution[j]:.3e}, "
                      f"B={B_distribution[j]:.3e}, sum={denom:.3e}")
                P_B_SC = 0.5
            else:
                P_B_SC = B_distribution[j] / denom

            ProbB_SC.append(P_B_SC)
            # Use a Bernoulli (binomial with n=1) distribution to make a choice, where 1 = Category B, 0 = Category A.
            intent_choice_SC = 1 if P_B_SC > 0.5 else 0

            if intent_choice_SC == 0:
                # Intended to choose A (0). 
                # Accidentally flips to B (1) with probability lambda_A.
                choice_SC = rng_SC.binomial(1, lambda_A)
            else:
                # Intended to choose B (1). 
                # Accidentally flips to A (0) with probability lambda_B.
                # This means it successfully STAYS B (1) with probability (1 - lambda_B).
                choice_SC = rng_SC.binomial(1, 1 - lambda_B)

            Response_SC.append(choice_SC)



            # Apply a Gaussian window centered at x[j] to update beliefs.
            g = norm.pdf(x, loc=x[j], scale=sigma_update)

            # Set a negative feedback coefficient (for incorrect choices).
            neg_gamma = gamma

            # Update belief distributions based on the subject's choice and feedback:

            # Case 1: Subject chose Category A (choice == 0) and the correct category is A.
            if choice_SC == 0 and categories[i] == 0:
                # Strengthen belief in Category A using positive feedback.
                A_distribution = A_distribution * gamma + g * (1 - gamma)
                # Normalize the belief distribution.
                A_distribution = A_distribution / trapezoid(A_distribution, x)

            # Case 2: Subject chose Category A but the correct category is B.
            elif choice_SC == 0 and categories[i] == 1:
                # Reduce belief in Category A using negative feedback.
                A_distribution = A_distribution * neg_gamma - g * (1 - neg_gamma)

                # Check for negative values in the distribution and shift it if necessary.
                min_yA = np.min(A_distribution)
                if min_yA < 0:
                    A_distribution = A_distribution + np.abs(min_yA)

                # Normalize the belief distribution.
                A_distribution = A_distribution / trapezoid(A_distribution, x)

            # Case 3: Subject chose Category B (choice == 1) and the correct category is B.
            elif choice_SC == 1 and categories[i] == 1:
                # Strengthen belief in Category B using positive feedback.
                B_distribution = B_distribution * gamma + g * (1 - gamma)
                # Normalize the belief distribution.
                B_distribution = B_distribution / trapezoid(B_distribution, x)

            # Case 4: Subject chose Category B but the correct category is A.
            elif choice_SC == 1 and categories[i] == 0:
                # Reduce belief in Category B using negative feedback.
                B_distribution = B_distribution * neg_gamma - g * (1 - neg_gamma)

                # Check for negative values in the distribution and shift it if necessary.
                min_yB = np.min(B_distribution)
                if min_yB < 0:
                    B_distribution = B_distribution + np.abs(min_yB)

                # Normalize the belief distribution.
                B_distribution = B_distribution / trapezoid(B_distribution, x)

            ####################################################################################################
            ######                                       BE part                                          ######
            ####################################################################################################

            # Calculate the cumulative probability (CDF) up to the closest x-value
            P_B_BE = calculate_cdf(x[j], x, y_BE)
            ProbB_BE.append(P_B_BE)     # Store the probability of choosing category B

            # Calculating subjects choice- 0 for A, 1 for B
            intent_choice_BE = 1 if P_B_BE > 0.5 else 0

            if intent_choice_BE == 0:
                # Intended to choose A (0). 
                # Accidentally flips to B (1) with probability lambda_A.
                choice_BE = rng_BE.binomial(1, lambda_A)
            else:
                # Intended to choose B (1). 
                # Accidentally flips to A (0) with probability lambda_B.
                # This means it successfully STAYS B (1) with probability (1 - lambda_B).
                choice_BE = rng_BE.binomial(1, 1 - lambda_B)

            Response_BE.append(choice_BE)     # Store the subject's response

            # Update the PDF by applying the learning rate and Delta_learning for the i-th trial
            y_BE = y_BE - eta_learning * Delta_learning[i, :]


            # Check if there are any negative values in the updated PDF and shift them upward if needed
            min_y = np.min(y_BE)
            if min_y < 0:
                y_BE = y_BE + np.abs(min_y)   # Ensure the PDF stays non-negative

            # Normalize the updated PDF to ensure its integral equals 1
            integral_y = trapezoid(y_BE, x)
            y_BE = y_BE / integral_y



            ####################################################################################################
            ######                                    integration part                                    ######
            ####################################################################################################

            # 1. Convex Combination of Beliefs
            Final_P_B = alpha*P_B_SC + (1-alpha)*P_B_BE

            # 2. Internal Choice (argmax: deterministic decision)
            # If the hybrid belief for B is greater than 0.5, intend to choose B (1). Else, A (0).
            Final_intended_choice = 1 if Final_P_B > 0.5 else 0

            # 3. Final Action (incorporating decision noise / asymmetric lapse rates)
            if Final_intended_choice == 0:
                # Intended to choose A (0). 
                # Accidentally flips to B (1) with probability lambda_A.
                Final_choice = rng_Final.binomial(1, lambda_A)
            else:
                # Intended to choose B (1). 
                # Accidentally flips to A (0) with probability lambda_B.
                # This means it successfully STAYS B (1) with probability (1 - lambda_B).
                Final_choice = rng_Final.binomial(1, 1 - lambda_B)


            Final_ProbB.append(Final_P_B)
            Final_Response.append(Final_choice)


        # If the subject did not respond, append NaN for the response.
        else:
            Final_Response.append(np.nan)
            Response_SC.append(np.nan)
            Response_BE.append(np.nan)
            Final_ProbB.append(np.nan)
            ProbB_SC.append(np.nan)
            ProbB_BE.append(np.nan)

    # Final belief distributions for Category A and B after all updates.
    yA = A_distribution
    yB = B_distribution


    # Calculate rewards: 1 for correct choice (category matches response), otherwise 0.
    # output np.nan for missed trials:
    rewards_SC = [1 if categories[i] == Response_SC[i] else (np.nan if np.isnan(Response_SC[i]) else 0) for i in range(len(categories))]
    rewards_BE = [1 if categories[i] == Response_BE[i] else (np.nan if np.isnan(Response_BE[i]) else 0) for i in range(len(categories))]
    Final_rewards =  [1 if categories[i] == Final_Response[i] else (np.nan if np.isnan(Final_Response[i]) else 0) for i in range(len(categories))]

    return Final_Response, Final_rewards, Final_ProbB, Response_SC, rewards_SC, ProbB_SC, Response_BE, rewards_BE, ProbB_BE, y_BE, yA, yB



def Hybrid_model(x, s_hat, categories, sigma_noise, A_repulsion, gamma, sigma_update, y_BE, Delta_learning, eta_learning, sigma_boundary, alpha,lambda_A, lambda_B, no_response,
                  seed, burn_in_seed, mode):
    


    if mode == 'simulated':

        # simulated mode: simulate expert behavior
        np.random.seed(burn_in_seed)
        m = 1000
        s_pre = np.random.uniform(low=-1, high=1, size=m)
        categories_pre = np.where(s_pre > 0, 1, 0)
        s_tilde_pre = s_pre + Noise_generator(m, seed, sigma_noise)
        s_hat_pre = Delta_repulsion(A_repulsion, s_tilde_pre)
        no_response_pre = np.full(m, False)

        # Concatenate simulated expert trials with actual trials
        s_hat = np.concatenate((s_hat_pre, s_hat))
        categories = np.concatenate((categories_pre, categories))
        no_response = np.concatenate((no_response_pre, no_response))

        # Learning and update
        Delta_learning_matrix = Delta_learning(x, s_hat, categories, sigma_boundary)

        Final_Response, Final_rewards, Final_ProbB, Response_SC, rewards_SC, ProbB_SC, Response_BE, rewards_BE, ProbB_BE, y_boundary, yA, yB = hybrid_calc(x, s_hat, categories, gamma, sigma_update, y_BE, Delta_learning_matrix, eta_learning, alpha,lambda_A, lambda_B, no_response, seed)

        # Exclude simulated trials from analysis
        Final_output_choice, Final_output_rewards = np.array(Final_Response[m:]) , np.array(Final_rewards[m:])
        SC_output_choice, SC_output_rewards = np.array(Response_SC[m:]), np.array(rewards_SC[m:])
        BE_output_choice, BE_output_rewards = np.array(Response_BE[m:]), np.array(rewards_BE[m:])
 


    elif mode =='real':
        # real mode: skip early real trials
        n_burn = 200
        # Learning and update
        Delta_learning_matrix = Delta_learning(x, s_hat, categories, sigma_boundary)

        Final_Response, Final_rewards, Final_ProbB, Response_SC, rewards_SC, ProbB_SC, Response_BE, rewards_BE, ProbB_BE, y_boundary, yA, yB = hybrid_calc(x, s_hat, categories, gamma, sigma_update, y_BE, Delta_learning_matrix, eta_learning, alpha,lambda_A, lambda_B, no_response, seed)

        # Exclude simulated trials from analysis
        Final_output_choice, Final_output_rewards = np.array(Final_Response[n_burn:]) , np.array(Final_rewards[n_burn:])
        SC_output_choice, SC_output_rewards = np.array(Response_SC[n_burn:]), np.array(rewards_SC[n_burn:])
        BE_output_choice, BE_output_rewards = np.array(Response_BE[n_burn:]), np.array(rewards_BE[n_burn:])

    else:
        raise ValueError("Invalid mode. Use 'simulated' or 'real'.")
    



    return Final_output_choice, Final_output_rewards, SC_output_choice, SC_output_rewards, BE_output_choice, BE_output_rewards





