#This file is the main entry point for Swarm Learning for MxNet
# platform. Users can integrate Swarm framework into their 
# model code by creating an instance of the SwarmCallback class and 
# calling its methods at different phases of training.

from __future__ import print_function

import mxnet
from swarmlearning.client.swarm import SwarmCallbackBase, SLPlatforms

# Default Training contract used for learning if not specified by user. 
# Any update to default contract needs similar modifications 
# in all applicable ML platforms (TF, PYT, MxNet, etc)
DEFAULT_TRAINING_CONTRACT = 'defaultbb.cqdb.sml.hpe'

class SwarmCallback(SwarmCallbackBase):
    '''
    This is the customized callback class sub-classed from 
    SwarmCallbackBase class that implements different swarm 
    functionalities. It implements the methods like 
    on_train_begin, on_batch_end etc and calls different 
    methods of SwarmCallbackBase.
    '''

    # Creating MxNet context
    # Put all artifacts needed to interface with ML platform here
    class MxNetContext:
        def __init__(self, model):
            self.model = model


    def __init__(self, syncFrequency, minPeers, trainingContract=DEFAULT_TRAINING_CONTRACT, **kwargs):
        '''
        This function initializes the various Swarm network parameters, which 
        are described below -
        :param syncFrequency: Batches of local training to be performed between 
                              2 swarm sync rounds. If adaptive sync enabled, this 
                              is the frequency to be used at the start.
        :param minPeers: Min peers required during each sync round for Swarm to 
                          proceed further.
        :param trainingContract: Training contract associated with this learning. 
                                 Default value is 'defaultbb.cqdb.sml.hpe'.
        :param useAdaptiveSync: Modulate the next interval length post each sync 
                                  round based on perf on validation data.
        :param adsValData: Validation dataset - (X,Y) tuple or generator - used 
                             for adaptive sync
        :param adsValBatch_size: Validation data batch size
        :param checkinModelOnTrainEnd: Indicates which model to check-in once 
                                           local model training ends at a node.
                                           Allowed values: ['inactive', 'snapshot', 
                                           'active']
        :param nodeWeightage: A number between 1-100 to indicate the relative 
                               importance of this node compared to others
        :param mlPlatform: 'MxNet' ML Platform
        :param model: MxNet model
        :param logger: Basic Python logger. SwarmCallback class will invoke info, 
                       debug and error methods of this logger to fulfil its need.
                       If no logger is passed, then SwarmCallback class will create 
                       its own logger from basic python logger. If required, user 
                       can get hold of this logger instance to change the log level 
                       as follows -
                       import logging
                       from swarmlearning.pyt import SwarmCallback                       
                       swCallback = SwarmCallback(syncFrequency=128, minPeers=3)
                       swCallback.logger.setLevel(logging.DEBUG)
        '''
        SwarmCallbackBase.__init__(self, syncFrequency, minPeers, trainingContract, kwargs)        
        self._verifyAndSetPlatformContext(kwargs)
        self._swarmInitialize()

    
    def on_train_begin(self):
        '''
        MxNet specific on_train_begin implementation
        '''
        self._swarmOnTrainBegin()
    

    def on_batch_end(self, batch=None):
        '''
        MxNet specific on_batch_end implementation
        '''
        self._swarmOnBatchEnd()


    def on_epoch_end(self, epoch=None):
        '''
        MxNet specific on_epoch_end implementation
        '''
        self._swarmOnEpochEnd()

    
    def on_train_end(self):
        '''
        MxNet specific on_train_end implementation
        '''
        self._swarmOnTrainEnd()


    def _verifyAndSetPlatformContext(self, params):
        '''
        MxNet specific implementation of abstract method
        _verifyAndSetPlatform in SwarmCallbackBase class.
        It is the verification and initialization code specific 
        to MxNet.
        '''
        ml_platform = params.get('ml_platform', SLPlatforms.MXNET.name)
        if ml_platform not in [SLPlatforms.MXNET.name]:
            self._logAndRaiseError("Invalid ml platform type: %s" % (ml_platform))
        self.mlPlatform = SLPlatforms[ml_platform]
        self.model = params.get('model', None)
        if self.model is None:
            self._logAndRaiseError("MxNet model is None")
        else:
            self.__setMLContext(mxnetModel=self.model)


    def _getValidationDataForAdaptiveSync(self, valData, valBatchSize):
        '''
        MxNet specific implementation of abstract method
        _getValidationDataForAdaptiveSync in SwarmCallbackBase class.
        Currently swarm does not support this for MxNet.
        '''
        self._logAndRaiseError("Adaptive sync for MxNet is not supported")


    def _saveModelWeightsToDict(self):
        '''
        MxNet specific implementation of abstract method
        _saveModelWeightsToDict in SwarmCallbackBase class.
        Saves the model passed to it inside its context, along with 
        the list of key weightNames of model's weights.
        This is later used in the loadModel function for loading the 
        updated set of weights as a flat dictionary
        '''
        inDict = {}
        self.weightNames = []
        model = self.mlCtx.model
        # in MxNet model weights are stored in a orderedDict 
        # hence we dont need to ensure ordering, it should work as is.
        for wTensor in model.save_parameters():
            # Hoewever weights are Tensors , we have change it to numpy types
            # wTensor is a str so we can use it as is.
            if (model.save_parameters()[wTensor].context.current_device=='gpu'):
                inDict[wTensor] = model.save_parameters()[wTensor].as_in_context(mxnet.cpu(0)).asnumpy()
            else:
                inDict[wTensor] = model.save_parameters()[wTensor].asnumpy()

            self.weightNames.append(wTensor)
        return inDict


    def _loadModelWeightsFromDict(self, inDict):
        '''
        MxNet specific implementation of abstract method
        _loadModelWeightsFromDict in SwarmCallbackBase class.
        This function in tightly intertwined with saveModelWeightstoDict 
        function, updating the same model that was passed to the last call 
        of the save model function. Hence please use carefully
        :param inDict: The flat model weights' dictionary to be loaded in the model
        :return: Nothing is returned, the saved model is updated in-place
        '''

        model = self.mlCtx.model
        tempDict = {}
        for k in self.weightNames:
            tempDict[k] = mxnet.nd.array(inDict[k])
        model.load_parameters(tempDict, strict=False)


    def _calculateLocalLoss(self):
        '''
        MxNet specific implementation of abstract method
        _calculateLocalLoss in SwarmCallbackBase class.
        '''
        # TBD : To be implemented later
        return 0


    def __setMLContext(self, **params):
        ctx = SwarmCallback.MxNetContext(params['mxnetModel'])
        self.logger.debug("Initialized MxNet context for Swarm")
        self.mlCtx = ctx
