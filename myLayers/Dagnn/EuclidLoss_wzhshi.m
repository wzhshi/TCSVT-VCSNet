classdef EuclidLoss_wzhshi < dagnn.ElementWise
  properties
    loss = 'Euclidian'
    idx = [];
  end

  properties (Transient)
    average = 0
    numAveraged = 0
    lastPred = []
    loss_zuo = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = (inputs{1} - inputs{2}(:,:,obj.idx,:)) .^ 2;
      outputs{1} = sum(outputs{1}(:)) / (size(inputs{1},1) * size(inputs{1},2));
      n = obj.numAveraged;  
      obj.average = (n * obj.average + gather(outputs{1})) / (n + 1) ;
      obj.numAveraged = n + 1;
      obj.lastPred = inputs{1};
      temp = outputs{1}/2;
      obj.loss_zuo = sum(temp(:)) / size(inputs{1},4);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
%       derInputs{1} = derOutputs{1} * (2 * (inputs{1} - inputs{2}));
        derInputs{1} = derOutputs{1} * ((inputs{1} - inputs{2}(:,:,obj.idx,:)));
      derInputs{2} = []; % labels
      derParams = {};
    end
    
    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1] ;
    end
    
    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end
    
    function obj = EuclidLoss_wzhshi(varargin)
      obj.load(varargin) ;      
    end
  end
end
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         