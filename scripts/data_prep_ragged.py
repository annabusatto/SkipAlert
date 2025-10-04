import pandas as pd
import scipy.io as sio
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from pathlib import Path

data_dir = "/uu/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction/Data/TimetoPVC"
save_dir = Path("/uu/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction/Data/Experiments-CleanedUp-Labeled")
save_dir.mkdir(exist_ok=True)
shortest_experiment = '22-10-19'
experiment_dates = ['19-01-17', '19-01-24', '19-01-31', '19-07-11', '19-07-18', '19-12-05', '19-12-12', '20-01-16', '20-01-23', '20-02-20', '20-02-27', '22-10-17', '22-10-19', '22-10-24', '22-10-26', '22-11-02', '22-11-07', '22-11-09']

# Experiment design dictionary (already provided in your code)
experiment_design = {
    '19-01-17': {
        "control": [],
        "intervention": [range(98, 192), range(208, 272), range(288, 343), range(361, 426), range(442, 519)],
        "recovery": [range(192, 208), range(272, 288), range(343, 361), range(426, 442)]
    },
    '19-01-24': {
        "control": [range(245, 309), range(317, 321)],
        "intervention": [range(96, 156), range(166, 233), range(321, 380), range(388, 477)],
        "recovery": [range(156, 166), range(233, 245), range(309, 317), range(380, 388)]
    },
    '19-01-31': {
        "control": [range(154, 168)],
        "intervention": [range(86, 146), range(168, 228), range(236, 298), range(374, 435), range(443, 465)],
        "recovery": [range(146, 154), range(228, 236), range(298, 374), range(435, 443)]
    },
    '19-07-11': {
        "control": [range(1, 9), range(55, 67), range(90, 93), range(108, 112), range(122, 128), range(150, 156)],
        "intervention": [range(9, 41), range(41, 43), range(43, 55), range(67, 90), range(93, 108), range(112, 122), range(128, 150), range(156, 172), range(172, 258)],
        "recovery": []
    },
    '19-07-18': {
        "control": [range(102, 111), range(183, 199), range(271, 275), range(349, 357), range(444, 460)],
        "intervention": [range(111, 171), range(199, 259), range(275, 337), range(357, 420), range(460, 569)],
        "recovery": [range(171, 183), range(259, 271), range(337, 349), range(420, 444)]
    },
    '19-12-05': {
        "control": [range(121, 125), range(205, 209), range(287, 291), range(386, 390)],
        "intervention": [range(125, 189), range(209, 271), range(291, 359)],
        "recovery": [range(189, 205), range(271, 287), range(365, 386)]
    },
    '19-12-12': {
        "control": [range(52, 60), range(133, 149), range(219, 227), range(309, 317), range(395, 403)],
        "intervention": [range(60, 120), range(149, 211), range(227, 297), range(317, 387), range(403, 453)],
        "recovery": [range(120, 133), range(211, 219), range(297, 309), range(387, 395), range(453, 469)]
    },
    '20-01-16': {
        "control": [range(101, 109), range(137, 146), range(175, 187), range(258, 272), range(343, 351)],
        "intervention": [range(109, 126), range(147, 163), range(187, 247), range(272, 332), range(351, 412)],
        "recovery": [range(126, 137), range(163, 175), range(247, 258), range(332, 343), range(412, 420)]
    },
    '20-01-23': {
        "control": [range(125, 133), range(210, 219), range(308, 325), range(395, 411), range(480, 489)],
        "intervention": [range(133, 194), range(219, 280), range(325, 385), range(411, 471), range(489, 551)],
        "recovery": [range(194, 210), range(280, 299), range(385, 395), range(471, 480), range(551, 559)]
    },
    '20-02-20': {
        "control": [range(184, 193), range(270, 278), range(324, 341)],
        "intervention": [range(193, 254), range(278, 311), range(341, 386)],
        "recovery": [range(254, 270), range(311, 324)]
    },
    '20-02-27': {
        "control": [range(161, 171), range(242, 251), range(321, 339), range(375, 393), range(432, 441)],
        "intervention": [range(171, 232), range(251, 311), range(339, 365), range(393, 421), range(441, 502)],
        "recovery": [range(232, 242), range(311, 321), range(365, 375), range(421, 432), range(502, 519)]
    },
    '22-10-17': {
        "control": [range(119, 124), range(198, 203), range(297, 308), range(382, 387), range(480, 485), range(559, 564)],
        "intervention": [range(124, 185), range(203, 284), range(308, 369), range(387, 469), range(485, 546), range(564, 646)],
        "recovery": [range(185, 198), range(284, 297), range(369, 382), range(469, 480), range(546, 559), range(646, 659)]
    },
    '22-10-19': {
        "control": [range(112, 117), range(191, 196)],
        "intervention": [range(117, 178), range(196, 278)],
        "recovery": [range(178, 191), range(278, 291)]
    },
    '22-10-24': {
        "control": [range(118, 123), range(197, 202), range(297, 302), range(396, 401)],
        "intervention": [range(123, 184), range(202, 284), range(302, 363), range(401, 443)],
        "recovery": [range(184, 197), range(284, 297), range(363, 396)]
    },
    '22-10-26': {
        "control": [range(118, 123), range(197, 202), range(297, 302), range(377, 382), range(477, 487), range(561, 566), range(644, 649), range(744, 749), range(843, 848)],
        "intervention": [range(123, 184), range(202, 284), range(302, 363), range(382, 464), range(487, 548), range(566, 629), range(649, 731), range(749, 843), range(848, 909)],
        "recovery": [range(184, 197), range(284, 297), range(363, 377), range(464, 477), range(548, 561), range(629, 644), range(731, 744), range(909, 914)]
    },
    '22-11-02': {
        "control": [range(115, 120), range(194, 199), range(273, 278), range(347, 352), range(447, 452), range(509, 514), range(588, 593), range(646, 651)],
        "intervention": [range(120, 181), range(199, 256), range(278, 334), range(352, 434), range(452, 493), range(514, 575), range(593, 633), range(651, 709)],
        "recovery": [range(181, 194), range(256, 273), range(334, 347), range(434, 447), range(493, 509), range(575, 588), range(633, 646), range(709, 719)]
    },
    '22-11-07': {
        "control": [range(112, 117), range(191, 196), range(287, 292), range(366, 371)],
        "intervention": [range(117, 178), range(196, 278), range(292, 353), range(371, 453)],
        "recovery": [range(178, 191), range(278, 287), range(353, 366), range(453, 466)]
    },
    '22-11-09': {
        "control": [range(113, 118), range(192, 197), range(291, 296), range(354, 359)],
        "intervention": [range(118, 179), range(197, 278), range(296, 341), range(359, 420)],
        "recovery": [range(179, 192), range(278, 291), range(342, 354), range(420, 424)]
    },
}

# Function to extract run number from Run_Name
def extract_run_number(run_name):
    # Example: Run0115-b1-ac-cs.mat -> 115
    run_part = run_name.split('-')[0]  # Get 'Run0115'
    run_number = int(run_part.replace('Run', ''))  # Remove 'Run' and convert to int
    return run_number

# Function to assign label based on run number and experiment design
def assign_label(run_number, experiment_date, experiment_design):
    design = experiment_design[experiment_date]
    
    # Check control ranges
    for r in design['control']:
        if run_number in r:
            return 'control'
    
    # Check intervention ranges
    for r in design['intervention']:
        if run_number in r:
            return 'intervention'
    
    # Check recovery ranges
    for r in design['recovery']:
        if run_number in r:
            return 'recovery'
    
    # If the run number doesn't fall into any range, return None or a default label
    return 'unknown'

for experiment_date in experiment_dates:
    # Initialize lists to store the data
    X_data = []
    y_data = []
    labels = []  # To store the labels for each X, y pair
    print(f"Processing {experiment_date}")
    
    # Load Excel file
    excel_file_path = os.path.join(data_dir, f"intervention_data_{experiment_date}.xlsx")
    excel_data = pd.read_excel(excel_file_path)
        
    # Load MATLAB files
    matlab_files_dir = os.path.join("/uu/sci.utah.edu/projects/CEG/Data/InSitu", experiment_date, "Data/PostProcessed")
    
    for n, row in excel_data.iterrows():
        # Extract run number from Run_Name
        run_number = extract_run_number(row['Run_Name'])
        
        # Assign label based on run number and experiment design
        label = assign_label(run_number, experiment_date, experiment_design)
        
        # Load MATLAB file
        matlab_file_data = sio.loadmat(os.path.join(matlab_files_dir, row['Run_Name']))
        
        # Extract the struct
        ts_data = matlab_file_data['ts'][0, 0]
        
        # Extract the potvals data from the struct
        potvals = ts_data['potvals']
        
        if potvals is not None:
            # Append the potvals data to X_data
            X_data.append(potvals)
            
            # Append the corresponding Time_to_Next_PVC value to y_data
            y_data.append(row['Time_to_Next_PVC'])
            
            # Append the label to labels list
            labels.append(label)
    
    # Transpose and concatenate the arrays
    transposed_arrays = [arr.T for arr in X_data]
    arr_len = [len(arr) for arr in transposed_arrays]
    
    # Concatenate all the beats together
    X_concatenated = np.concatenate(transposed_arrays, axis=0)
    
    # Convert labels to a numpy array
    labels = np.array(labels)
    
    # Save the .npz file for the experiment, including the labels
    np.savez(save_dir / f"{experiment_date}.npz", 
             potvals=X_concatenated, 
             time_to_pvc=y_data, 
             arr_len=arr_len, 
             labels=labels)

    print(f"Saved data for {experiment_date} with labels: {np.unique(labels)}")