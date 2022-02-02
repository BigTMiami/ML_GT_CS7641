results = [
    ["Decision Tree", "Census", "ccp_alpha = 0.0000153", 95.30983741304304, 1.5305140018463135, 0.015937089920043945],
    ["Decision Tree", "MNIST", "ccp_alpha = 0.0000762", 88.31, 15.921523332595825, 0.006710052490234375],
    [
        "Neural Network",
        "Census",
        "network_learning_rate = 0.000001, layer_size = 16, epoch_count = 275",
        95.09332210661374,
        472.21750926971436,
        0.021795988082885742,
    ],
    [
        "Neural Network",
        "MNIST",
        "use_cnn=True, epoch_count=100",
        tensor(98.9600),
        1081.4384100437164,
        0.7877399921417236,
    ],
    [
        "Boosting ",
        "Census",
        "n_estimators=200, max_depth=3, max_features=5",
        9001074.16852108,
        11.063764095306396,
        0.2702629566192627,
    ],
    [
        "Boosting ",
        "MNIST",
        "n_estimators=25, max_depth=15, max_features=38",
        94.04,
        122.70928597450256,
        0.27556800842285156,
    ],
    [
        "Boosting ",
        "Census",
        "n_estimators=200, max_depth=3, max_features=5",
        95.57947916040176,
        11.18084192276001,
        0.2826659679412842,
    ],
    ["Support Vector Machine ", "Census", "kernel=poly", 94.83370421603416, 1768.4988389015198, 86.27147483825684],
]
