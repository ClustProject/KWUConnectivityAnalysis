a = load('C:\Users\daehyeon\Desktop\GCN_implement\ExtractedFeaturesDE_STFT_LDS_v2\dujingcheng_20131027.mat');
data = struct2cell(a);

label = [1	0	-1	-1	0	1	-1	0	1	1	0	-1	0	1	-1];
feature_order = [1, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5, 6, 7, 8 ,9];
k = 1;
for i=1:15  % Subject [1, 15]
    var_name = strcat('DE_STFT_LDS', num2str(feature_order(k)));
    if label(feature_order(k)) == 1
        fprintf('Positive');
    elseif label(feature_order(k)) == 0
        fprintf('Neutral');
    else
        fprintf('Negative');
    end
    k=k+1;
    file_path = strcat('./Figures/', var_name, '/');
    d = cell2mat(data(i));
    dlen = size(d,2);
    fig_path = strcat(file_path,var_name);
    mkdir(fig_path)
    for j = 1 : dlen
        hold;
        plot(d(:,j,1));
        hold on
        plot(d(:,j,2));
        plot(d(:,j,3));
        plot(d(:,j,4));
        plot(d(:,j,5));
        title(strcat('SUB1 - ', var_name, ' - One Sample (62 x 5)'))
        xlabel("The number of Nodes")
        ylabel("DE LDS Value")
        legend('Delta', '''Theta''', '''Alpha''', '''Beta''', '''Gamma''');
        
        fig_name = strcat(fig_path,'_Sample_',num2str(j));
        print(fig_name,'-dpng');
    end
end