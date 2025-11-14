file = 'C:\Users\User\Desktop\DSP MAMAW\DSP-MAMAW\data\ppg_dalia_all_activities_data.mat';
whos('-file', file)

S = load(file);
rec = S.data(1);      % inspect subject 1
fieldnames(rec)
