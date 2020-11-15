import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from pathlib import Path


# Params from question
a_min = 0
a_max = 15
b_min = 0
b_max = 200
# therefore..
c_min = a_min + b_min # 0
c_max = a_max + b_max # 215

# Factors from given probabilities in question
fa = p_a = 0.99
fb = p_b = 0.3
pd = 0.5
fd = 1 + pd

# therefore..
d_max = c_max * fd # 322.5

# Observations
D = np.array([22, 27, 26, 32, 31, 25, 35, 26, 28, 23])

def generate_cn_given_d(a_max, b_max, total_fans, pd):
    '''
    :param a_max: the max value of the model for a
    :param b_max: the max value of the model for b
    :param total_fans: the observed value of the number of fans at a game
    :param pd: the probability a ticket holder brings a family member
    :return: distribution of Cn, distribution of Cn (normalized)

    Example of how we work out distribution of Cn when we observe 10 fans:
    min_ticket holders = 5 (since we need at least half of the individuals to be ticket holders to make 10 fans)
    max_ticket holders = 10 (if everyone at the game was a ticket holder)
    p(cn = 1 | dn = 10 ) = 0 (since if only 1 person is a ticket holder at the game n, we cannot observe 10 people
    p(cn = 5 | dn = 10) = 5C5 * 0.5^5 * 0.5^0 (if every ticket holder brought a family)
    p(cn = 6 | dn = 10) = 6C4 * 0.5^4 * 0.5^2 (if there are 6 ticket holders and 10 fans were observed, 4 of them brough family)
    ...
    As we continue we build a new distribution cn where its 0 everywhere except between 5 and 10.
    '''
    min_ticket_holders = np.floor(total_fans / 2) # Since a ticket holder can bring only one, we can only at most double
    max_ticket_holders = total_fans

    #Create an array to store the Cn distribution (a_max+b_max+1) x 1 dimensions
    cn_distribution = np.zeros(a_max+b_max+1) #since we include 0, we need to do +1 to the sum
    non_zero_range = np.arange(min_ticket_holders, max_ticket_holders+1, 1)

    for num_of_ticket_holders in non_zero_range:
        num_with_family = total_fans - num_of_ticket_holders
        num_without_family = total_fans - num_with_family
        probability = comb(num_of_ticket_holders, num_with_family) * pd**num_with_family * (1-pd)**num_without_family
        cn_distribution[int(num_of_ticket_holders)] = probability

    return cn_distribution

def generate_a(a_min, a_max):
    a = np.ones((a_max-a_min)+1) * (1 / (a_max+1))
    return a

def generate_b(b_min, b_max):
    b = np.ones((b_max-b_min)+1) * (1 / (b_max+1))
    return b

def generate_c_given_a_b(a_max, b_max, pa, pb):



    a_range = np.arange(0, a_max+1, 1)
    b_range = np.arange(0, b_max+1, 1)

    c_given_a_b = np.zeros((a_max+1, b_max+1, a_max+b_max+1))

    for a in a_range:
        for b in b_range:
            c_max = a + b  # This is the max number of ticket holders at a match
            cmax_range = np.arange(0, c_max+1, 1)
            for c in cmax_range:
                an_range = np.arange(0, a+1, 1)  # an is the number of seasonal ticket holders that attended
                for an in an_range:
                    if an > c:
                        break
                    bn = c - an
                    # We now calculate the probability of c people attending
                    # an is the number of seasonal ticket holders going
                    # bn is the number of on-day ticket holders going
                    prob_an_out_of_a_going = comb(a, an) * pa**an * (1-pa)**(a-an)
                    prob_bn_out_of_b_going = comb(b, bn) * pb**bn * (1-pb)**(b-bn)
                    prob_an_bn = prob_an_out_of_a_going * prob_bn_out_of_b_going
                    c_given_a_b[a, b, c] = c_given_a_b[a, b, c] + prob_an_bn

    return c_given_a_b

def generate_conclusion_c(d_observations, pd, a_max, b_max):
    c_distribution = np.zeros(a_max+b_max+1)
    for d in d_observations:
        c_distribution = c_distribution + generate_cn_given_d(a_max, b_max, d, pd)

    return c_distribution



c_given_a_b = generate_c_given_a_b(a_max, b_max, p_a, p_b)
c_given_a_b_for_a = c_given_a_b
c_given_a_b_for_b = c_given_a_b
c = generate_conclusion_c(D, pd, a_max, b_max)


# Multiply the joint c's against every c column in c_given_a_b
a_range = np.arange(0, a_max+1, 1)
b_range = np.arange(0, b_max+1, 1)
c_range = np.arange(0, a_max+b_max+1, 1)

# Generating the a_posterior
for a in a_range:
    for b in b_range:
        c_given_a_b_for_a[a, b, :] = c_given_a_b_for_a[a, b, :] * c
        c_given_a_b_for_a[a, b, :] = c_given_a_b_for_a[a, b, :] * (1 / b_max+1)

a_given_b = np.zeros((a_max+1, b_max+1))
for cn in c_range:
    a_given_b = a_given_b + c_given_a_b[:, :, cn]

posterior_a = np.zeros(a_max+1)
for b in b_range:
    posterior_a = posterior_a + a_given_b[:, b]

# Generating the b_posterior
for a in a_range:
    for b in b_range:
        c_given_a_b_for_b[a, b, :] = c_given_a_b_for_b[a, b, :] * c
        c_given_a_b_for_b[a, b, :] = c_given_a_b_for_b[a, b, :] * (1 / a_max+1)

b_given_a = np.zeros((a_max+1, b_max+1))
for cn in c_range:
    b_given_a = b_given_a + c_given_a_b_for_b[:, :, cn]

posterior_b = np.zeros(b_max+1)
for a in a_range:
    posterior_b = posterior_b + b_given_a[a, :]

plt.figure()
posterior_a_normalized = posterior_a / sum(posterior_a)
posterior_b_normalized = posterior_b / sum(posterior_b)
plt.subplot(211)
plt.bar(a_range, posterior_a_normalized)
plt.subplot(212)
plt.bar(b_range, posterior_b_normalized)
plt.show()

print("Ding!")



