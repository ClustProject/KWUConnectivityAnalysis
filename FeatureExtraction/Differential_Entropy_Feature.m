
subject_file_struct = dir('C:\Users\daehyeon\Desktop\GCN_implement\GCN_implement\dataset\SEED\SEED\Preprocessed_EEG\data');
sub_len = size(subject_file_struct,1)-2;
trial_len = 15;
num_nodes = 62;
num_frequency_bands = 5;
hann_window_size = 200;
fft_length = 512;
overlap_length = 0;
bandwidth = [1 4 8 14 31; 3 7 13 30 50];

fs = 200;
n = 8;

for i=1:sub_len%sub_len  % Subject [1, 45]
    file_path = strcat(subject_file_struct(i+2).folder,'\',subject_file_struct(i+2).name);
    data = load(file_path);
fprintf("Run : {%d}\n", i);
    all_trial_data = struct2cell(data);
    
    file_name = strcat('ExtractedFeaturesDE_STFT', '\', subject_file_struct(i+2).name);
    
    trials_name_list = [];
    for j=1:trial_len % Trial [1, 15]
        trial_data = cell2mat(all_trial_data(j));
        trial_EEG_len = size(trial_data, 2);
    
        num_EEG_blocks = int32(trial_EEG_len / fs);
    
        trial_features = zeros(num_nodes, num_EEG_blocks, num_frequency_bands);
        for row=1:num_nodes % Nodes = 62
            EEG_data = trial_data(row,:);
    
            %extracted_waves = brain_waves(EEG_data,fs, n);
            
            %wave = extracted_waves(w,:);
            %fprintf("%d, %d\n",col,window_len);
            %window = wave(window_len:window_len+fs-1);
            [s,f] = STFT(EEG_data, fs, hann_window_size, fft_length, overlap_length);
            de = DE(s,f,num_frequency_bands,bandwidth); % approximately 235 x 5
            
            %fprintf("Waves Feature Size: %d x %d\n", size(all_waves_features, 1), size(all_waves_features, 2));
            trial_features(row,:,:) = de;
        end
        %fprintf("Trial Feature Size: %d x %d x %d\n", size(trial_features, 1), size(trial_features, 2), size(trial_features, 3));
        %각 Trial feature들을 변수로 저장해야함

        renamed = strcat('DE',num2str(j));
        eval([genvarname(renamed), '= trial_features;']);
        renamed = strcat("DE",string(j));
        trials_name_list = [trials_name_list; renamed];
    end
    a2 = char(trials_name_list(1)); b2 = char(trials_name_list(2)); c2 = char(trials_name_list(3));
    d2 = char(trials_name_list(4));e2 = char(trials_name_list(5));f2 = char(trials_name_list(6));
    g2 = char(trials_name_list(7));h2 = char(trials_name_list(8));i2 = char(trials_name_list(9));
    j2 = char(trials_name_list(10));k2 = char(trials_name_list(11));l2 = char(trials_name_list(12));
    m2 = char(trials_name_list(13));n2 = char(trials_name_list(14));o2 = char(trials_name_list(15));
    save(file_name, a2,b2,c2,d2,e2,f2,g2,h2,i2,j2,k2,l2,m2,n2,o2);
end



        

