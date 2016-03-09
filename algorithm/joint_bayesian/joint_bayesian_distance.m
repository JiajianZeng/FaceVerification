function [ distance ] = joint_bayesian_distance( model, x1, x2 )
%Compute the joint bayesian distance between x1 and x2 according to the
%parameter of the model
%   model holds the parameters of the joint bayesian model
distance = x1 * model.A * x1' + x2 * model.A * x2' - 2 * x1 * model.G * x2';
end

