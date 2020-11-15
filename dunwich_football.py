import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

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

def generate_bn_from_b(b_min, b_max, fb):
    b_range = (b_max - b_min) + 1  # Since we include 0 as well
    bn_b_joint_distribution = np.zeros((b_range, b_range))  # For max_b = 200, we should expect a 201 x 201 matrix
    b_distribution = np.ones(b_range) * (1 / b_range) #uniform distribution
    possible_b_values = np.arange(0, b_range, 1)

    for b_val in possible_b_values:
        possible_bn_values = np.arange(0, b_val+1, 1)  #We can be between 0
        bn_distribution = np.zeros(b_range)
        for bn_val in possible_bn_values:
            probability = comb(b_val, bn_val) * fb**bn_val * (1-fb) ** (b_val - bn_val)
            log_prob = abs(np.log(probability))
            bn_distribution[int(bn_val)] = log_prob
        bn_b_joint_distribution[int(b_val)] = bn_distribution

    bn_marginal = np.zeros(b_range)
    for b_val in possible_b_values:
        bn_marginal = bn_marginal + bn_b_joint_distribution[b_val]

    bn_marginal_normalized = bn_marginal / sum(bn_marginal)
    return bn_marginal, bn_marginal_normalized, bn_b_joint_distribution

def generate_bn_joint_cn_given_d(bn, cn):
    cn_dim = cn.shape[0]
    bn_dim = bn.shape[0]
    bn_extended = np.zeros(cn_dim)
    bn_extended[0:bn_dim] = bn
    bn_extended = bn_extended.reshape((cn_dim, 1))
    cn_shaped = cn.reshape((cn_dim, 1)).T
    bn_joint_cn_given_d = bn_extended @ cn_shaped
    plt.imshow(bn_joint_cn_given_d)
    plt.show()
    return bn_joint_cn_given_d

def generate_an_or_bn(max, attendance_probability):
    bn_values = np.arange(0, max+1, 1)
    b_values = np.arange(0, max+1, 1)
    bn_given_b = np.zeros((max+1, max+1))
    for bn in bn_values:
        for b in b_values:
            if bn > b:
                bn_given_b[bn, b] = 0
            else:
                bn_given_b[bn, b] = comb(b, bn) * (attendance_probability)**bn * (1-attendance_probability)**(b - bn)

    # Find the marginal of bn
    plt.figure()
    plt.imshow(bn_given_b)
    plt.show()
    b_prior = np.ones(max+1) * (1 / max)
    bn_given_b_times_b = bn_given_b * (1/max)
    bn_marginal = np.zeros(max+1)
    for b in b_values:
        slice_of_joint = bn_given_b_times_b[:, b]
        bn_marginal = bn_marginal + slice_of_joint

    return bn_marginal

def generate_an_given_d(bn, cn, an_max):
    cn_dim = cn.shape[0]
    bn_dim = bn.shape[0]
    bn_extended = np.zeros(cn_dim)
    bn_extended[0:bn_dim] = bn
    bn_extended = bn_extended.reshape((cn_dim, 1))
    cn_shaped = cn.reshape((cn_dim, 1)).T
    joint_cn_bn = bn_extended @ cn_shaped
    an_given_d = np.zeros(16)  # 0 to 15
    '''
    Calculate the posterior of an. We need to establish the constraints 
    1. index of bn cannot be greater than the index cn, i.e. you cannot have 6 normal fans when there were only 4 ticket
       holders at the game cn 
    2. index cn cannot be greater than index of bn by the max possible value of an 
    '''
    for bn_row in range(cn_dim):
        for cn_column in range(cn_dim):
            if bn_row > cn_column:
                continue
            elif cn_column - bn_row > an_max:
                continue
            else:
                an_index = cn_column - bn_row
                an_given_d[an_index] = an_given_d[an_index] + joint_cn_bn[bn_row, cn_column]

    return an_given_d

def generate_a_given_an(a_max, pa):
    a_given_an = np.zeros((a_max+1, a_max+1))
    an_values = np.arange(0, a_max+1, 1)
    a_values = np.arange(0, a_max+1, 1)
    for a in a_values:
        for an in an_values:
            if an > a:
                a_given_an[a, an] = 0
            else:
                a_given_an[a, an] = comb(a, an) * (pa ** an) * ((1-pa) ** (a - an))

    return a_given_an

'''
Calcualting the posterior for a

1. Calculate p(bn) 
2. Calculate p(cn | dn = d1)
3. compute the likelihood p(an | dn = d1) 
4. compute the likelihood p(a | an) 
5. compute the posterior p(a) = sum {an} p(a | an) * p(an | dn = 1)
'''
cn_given_d = generate_cn_given_d(a_max, b_max, D[0], pd)
bn = generate_an_or_bn(b_max, p_b)
an_given_d = generate_an_given_d(bn, cn_given_d, a_max)
a_given_an = generate_a_given_an(a_max, p_a)
posterior_a = a_given_an @ an_given_d

'''
Calculating the posterior for b 
1. Calculate p(an)
2. Calculate p(cn | dn = d1) 
3. Compute the likelihood p(b | bn) 
4. compute the likelihood p(b) = sum {an} p(a | an) * p(an | dn = 1)
'''