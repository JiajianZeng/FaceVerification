function [ ] = svm_model_2_yaml( svm_model, filename )
%convert svm_model to yaml file

% initialize
Parameters = svm_model.Parameters;
nr_class = svm_model.nr_class;
totalSV = svm_model.totalSV;
rho = svm_model.rho;
Label = svm_model.Label;
sv_indices = svm_model.sv_indices;
ProbA = svm_model.ProbA;
ProbB = svm_model.ProbB;
nSV = svm_model.nSV;
sv_coef = svm_model.sv_coef;
SVs = svm_model.SVs;

% write Parameters
mat2yaml(Parameters, 'Parameters', 'f', filename, 'w');
% write nr_class
mat2yaml(nr_class, 'nr_class', 'f', filename, 'a');
% write totalSv
mat2yaml(totalSV, 'totalSV', 'f', filename, 'a');
% write rho
mat2yaml(rho, 'rho', 'f', filename, 'a');
% write Label
mat2yaml(Label, 'Label', 'f', filename, 'a');
% write sv_indices
mat2yaml(sv_indices, 'sv_indices', 'f', filename, 'a');
% write ProbA
mat2yaml(ProbA, 'ProbA', 'f', filename, 'a');
% write ProbB
mat2yaml(ProbB, 'ProbB', 'f', filename, 'a');
% write nSV
mat2yaml(nSV, 'nSV', 'f', filename, 'a');
% write sv_coef
mat2yaml(sv_coef, 'sv_coef', 'f', filename, 'a');
% write SVs sparse matrix
mat2yaml(full(SVs), 'SVs', 'f', filename, 'a');

        
end

