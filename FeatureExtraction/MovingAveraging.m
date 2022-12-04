subject_file_struct = dir('C:\Users\daehyeon\Desktop\GCN_implement\ExtractedFeaturesDE_STFT');
sub_len = size(subject_file_struct,1)-2;
trial_len = 15;
num_nodes = 62;
num_frequency_bands = 5;
window_len = 20;
half_window = int8(window_len/2);

fs = 200;
n = 8;

feature_order = [1, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5, 6, 7, 8 ,9];
for i=1:sub_len  % Subject [1, 45]
    file_path = strcat(subject_file_struct(i+2).folder,'\',subject_file_struct(i+2).name);
    data = load(file_path);

    all_trial_data = struct2cell(data);
    
    file_name = strcat('ExtractedFeaturesDE_STFT_MA', '\', subject_file_struct(i+2).name);
    
    trials_name_list = [];
    for j=1:trial_len % Trial [1, 15]
        trial_feature = cell2mat(all_trial_data(j));
        feature_len = size(trial_feature, 2);
        
        MA_val = zeros(num_frequency_bands);
        MA_trial_feature = zeros(num_nodes, feature_len, num_frequency_bands);
        left_padd_end = half_window - 1;
        right_padd_start = int16(half_window)+feature_len;
        for row=1:num_nodes % Nodes = 62
            node_feature = trial_feature(row,:,:); % dim 1
            % 앞 9, 뒤 10 Same Padding 하기
            padding_node_feature = zeros(feature_len+window_len-1, num_frequency_bands);

            for t = 1 : left_padd_end
                padding_node_feature(t,:) = transpose(squeeze(node_feature(1, 1, :)));
            end
            padding_node_feature(int16(half_window):right_padd_start-1,:) = squeeze(node_feature);
            for t = right_padd_start : right_padd_start + int16(half_window)
                padding_node_feature(t,:) = transpose(squeeze(node_feature(1, feature_len, :)));
            end

            for w = 1:feature_len
                MA_val = mean(padding_node_feature(w:w+window_len-1, :), 1); % dim 1로 평균
                MA_trial_feature(row,w,:) = MA_val;
            end
        end
        renamed = strcat('DE_MA',num2str(feature_order(j)));
        eval([genvarname(renamed), '= MA_trial_feature;']);
        renamed = strcat("DE_MA",num2str(feature_order(j)));
        trials_name_list = [trials_name_list; renamed];
    end
    a2 = char(trials_name_list(1)); b2 = char(trials_name_list(2)); c2 = char(trials_name_list(3));
    d2 = char(trials_name_list(4));e2 = char(trials_name_list(5));f2 = char(trials_name_list(6));
    g2 = char(trials_name_list(7));h2 = char(trials_name_list(8));i2 = char(trials_name_list(9));
    j2 = char(trials_name_list(10));k2 = char(trials_name_list(11));l2 = char(trials_name_list(12));
    m2 = char(trials_name_list(13));n2 = char(trials_name_list(14));o2 = char(trials_name_list(15));
    save(file_name, a2,b2,c2,d2,e2,f2,g2,h2,i2,j2,k2,l2,m2,n2,o2);
end
