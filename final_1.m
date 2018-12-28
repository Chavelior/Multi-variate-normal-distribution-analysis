training_part_F1 = F1(1:100,1:5);
training_part_F2 = F2(1:100,1:5);

[m11,C11] = normfit(training_part_F1(1:100,1));
[m12,C12] = normfit(training_part_F1(1:100,2));
[m13,C13] = normfit(training_part_F1(1:100,3));
[m14,C14] = normfit(training_part_F1(1:100,4));
[m15,C15] = normfit(training_part_F1(1:100,5));

[m21,C21] = normfit(training_part_F2(1:100,1));
[m22,C22] = normfit(training_part_F2(1:100,2));
[m23,C23] = normfit(training_part_F2(1:100,3));
[m24,C24] = normfit(training_part_F2(1:100,4));
[m25,C25] = normfit(training_part_F2(1:100,5));

% Measure the probability that X belongs

F1_pred = [];
F1_real = [];
Z1_pred = [];
F2_pred = [];
Z1F2_pred = [];
F2Z1_pred = []; % dummy
for i = 101:1000
    for j = 1:5
        x = F1(i,j);
        y = F2(i,j);
        C1_val_F1 = normpdf(x,m11,C11)*(2/5);
        C2_val_F1 = normpdf(x,m12,C12)*(2/5);
        C3_val_F1 = normpdf(x,m13,C13)*(2/5);
        C4_val_F1 = normpdf(x,m14,C14)*(2/5);
        C5_val_F1 = normpdf(x,m15,C15)*(2/5); 
        class_pred_F1 = find([C1_val_F1,C2_val_F1,C3_val_F1,C4_val_F1,C5_val_F1] == max([C1_val_F1,C2_val_F1,C3_val_F1,C4_val_F1,C5_val_F1]));
        
        C1_val_Z1 = normpdf((x-m11)/C11)*(2/5);
        C2_val_Z1 = normpdf((x-m12)/C12)*(2/5);
        C3_val_Z1 = normpdf((x-m13)/C13)*(2/5);
        C4_val_Z1 = normpdf((x-m14)/C14)*(2/5);
        C5_val_Z1 = normpdf((x-m15)/C15)*(2/5);
        class_pred_Z1 = find([C1_val_Z1,C2_val_Z1,C3_val_Z1,C4_val_Z1,C5_val_Z1] == max([C1_val_Z1,C2_val_Z1,C3_val_Z1,C4_val_Z1,C5_val_Z1]));
        
        C1_val_F2 = normpdf(x,m21,C21)*(2/5);
        C2_val_F2 = normpdf(x,m22,C22)*(2/5);
        C3_val_F2 = normpdf(x,m23,C23)*(2/5);
        C4_val_F2 = normpdf(x,m24,C24)*(2/5);
        C5_val_F2 = normpdf(x,m25,C25)*(2/5); 
        class_pred_F2 = find([C1_val_F2,C2_val_F2,C3_val_F2,C4_val_F2,C5_val_F2] == max([C1_val_F2,C2_val_F2,C3_val_F2,C4_val_F2,C5_val_F2]));
        
        C1_val_F2Z1 = mvnpdf([((x - m11)/C11) y], [0 m21],[1 C21]);
        C2_val_F2Z1 = mvnpdf([((x - m12)/C12) y], [0 m22],[1 C22]);
        C3_val_F2Z1 = mvnpdf([((x - m13)/C13) y], [0 m23],[1 C23]);
        C4_val_F2Z1 = mvnpdf([((x - m14)/C14) y], [0 m24],[1 C24]);
        C5_val_F2Z1 = mvnpdf([((x - m15)/C15) y], [0 m25],[1 C25]);
        class_pred_F2Z1 = find([C1_val_F2Z1,C2_val_F2Z1,C3_val_F2Z1,C4_val_F2Z1,C5_val_F2Z1] == max([C1_val_F2Z1,C2_val_F2Z1,C3_val_F2Z1,C4_val_F2Z1,C5_val_F2Z1]));
        
        F1_real = [F1_real j];
        F1_pred = [F1_pred class_pred_F1];
        F2_pred = [F2_pred class_pred_F2];
        Z1_pred = [Z1_pred class_pred_Z1];
%        Z1F2_pred = [Z1F2_pred class_pred_Z1F2];
        F2Z1_pred = [F2Z1_pred class_pred_F2Z1];
    end
end

accur_F1 = [];
accur_F2 = [];
accur_Z1 = [];
accur_Z1F2 = [];
accur_F2Z1 = [];
for i = 1:4500
    if F1_real(i) == F1_pred(i)
        accur_F1(i) = 1;
    else
        accur_F1(i) = 0;
    end
    if F1_real(i) == Z1_pred(i)
        accur_Z1(i) = 1;
    else
        accur_Z1(i) = 0;
    end
    if F1_real(i) == F2_pred(i)
        accur_F2(i) = 1;
    else
        accur_F2(i) = 0;
    end
    if F1_real(i) == F2Z1_pred(i)
        accur_F2Z1(i) = 1;
    else
        accur_F2Z1(i) = 0;
    end
end


Classification_accuracy_F1 = sum(accur_F1)/4500; %0.53
Error_rate_F1 = (4500 - sum(accur_F1))/4500 %0.47
Classification_accuracy_Z1 = sum(accur_Z1)/4500; %0.5262
Error_rate_Z1 = (4500 - sum(accur_Z1))/4500 %0.4738
Classification_accuracy_F2 = sum(accur_F2)/4500 %0.2144
Error_rate_F2 = (4500 - sum(accur_F2))/4500; %0.7856
Classification_accuracy_F2Z1 = sum(accur_F2Z1)/4500 %0.7913
Error_rate_F2Z1 = (4500 - sum(accur_F2Z1))/4500; %0.2087

Z11 = [];
Z12 = [];
Z13 = [];
Z14 = [];
Z15 = [];
Y1 = [];
Y2 = [];
Y3 = [];
Y4 = [];
Y5 = [];

for i = 1:1000
        x1 = F1(i,1);
        x2 = F1(i,2);
        x3 = F1(i,3);
        x4 = F1(i,4);
        x5 = F1(i,5);
        y1 = F2(i,1);
        y2 = F2(i,2);
        y3 = F2(i,3);
        y4 = F2(i,4);
        y5 = F2(i,5);
        Z11_app = (x1-m11)/C11;
        Z11 = [Z11 Z11_app];
        Z12_app = (x2-m12)/C12;
        Z12 = [Z12 Z12_app];
        Z13_app = (x3-m13)/C13;
        Z13 = [Z13 Z13_app];
        Z14_app = (x4-m14)/C14;
        Z14 = [Z14 Z14_app];
        Z15_app = (x5-m15)/C15;
        Z15 = [Z15 Z15_app];
        
        Y1 = [Y1 y1];
        Y2 = [Y2 y2];        
        Y3 = [Y3 y3];
        Y4 = [Y4 y4];
        Y5 = [Y5 y5];        
        
        Z1 = [Z11 Z12 Z13 Z14 Z15];
        F2_plot = [Y1 Y2 Y3 Y4 Y5];
end

scatter(Z11,Y1,'r');
hold on
scatter(Z12,Y2,'b');
hold on
scatter(Z13,Y3,'g');
hold on
scatter(Z14,Y4,'y');
hold on
scatter(Z15,Y5,'k');
xlabel('Z1');
ylabel('F2');     
legend('C1','C2','C3','C4','C5');




