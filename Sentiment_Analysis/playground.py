outlook=[
    {"sunny":{"yes":2, "no":3}},
    {"overcast":{"yes":4, "no":0}},
    {"rainy":{"yes":3, "no":2}},
    ]

P_outlook_yes={"sunny":2/9, "overcast":4/9, "rainy":3/9}
P_outlook_no={"sunny":3/5, "overcast":0, "rainy":2/5}

temp=[
    {"hot":{"yes":2, "no":2}},
    {"mild":{"yes":4, "no":2}},
    {"cool":{"yes":3, "no":1}},
    ]
P_temp_yes={"hot":2/9, "mild":4/9, "cool":3/9}
P_temp_no={"hot":2/5, "mild":2/5, "cool":1/5}

play=[{"yes":9, "no":5}]
P_play_yes=9/14
P_play_no=5/14

train_samp=[
    ("sunny","hot"),
    ("sunny","mild"),
    ("sunny","cool")
    ]

def train(sample):
    #p(yes/today)= p(sunny/yes) * p(hot/yes) * p(yes)
    P_yes= P_outlook_yes[sample[0]] * P_temp_yes[sample[1]] * P_play_yes
    P_no= P_outlook_no[sample[0]] * P_temp_no[sample[1]] * P_play_no

    output_sum= P_yes + P_no
    norm_output= {"yes":P_yes/output_sum, "no":P_no/output_sum}

    return norm_output

train_outputs= [train(samp) for samp in train_samp]

for result in train_outputs:
    print(result)
