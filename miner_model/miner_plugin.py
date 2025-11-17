# The MIT License (MIT)


"""
Bittbridge Miner Plugin

This is the main miner file that integrates your predictive model with the Bittensor network.
The miner handles network communication, while your model handles predictions.

To use this miner:
1. Implement a model that inherits from PredictionModel (see model_interface.py)
2. Instantiate your model
3. Pass it to the Miner class
4. Run the miner

Example:
    from example_models.simple_model import SimpleAPIModel
    from miner_plugin import Miner
    
    model = SimpleAPIModel()
    miner = Miner(config=None, model=model)
    miner.run()
"""

import time
import typing
import bittensor as bt

# Bittensor Miner Template:
import bittbridge

# Import base miner class which takes care of most of the boilerplate
from bittbridge.base.miner import BaseMinerNeuron

# Import the model interface and example model
from .model_interface import PredictionModel
# Import the simple example model, this placeholder should be changed with your own model
from .example_models.simple_model import SimpleAPIModel


class Miner(BaseMinerNeuron):
    """
    Miner neuron class that integrates your predictive model with the Bittensor network.
    
    This class handles:
    - Network communication (receiving challenges from validators)
    - Request filtering (blacklist/priority)
    - Delegating predictions to your model
    
    To customize:
    1. Replace the model with your own (see __init__)
    2. The forward() method delegates to your model - no changes needed
    
    This class inherits from BaseMinerNeuron, which handles:
    - Wallet and subtensor setup
    - Metagraph synchronization
    - Logging configuration
    - Axon (server) setup
    """

    def __init__(self, config=None, model: PredictionModel = None):
        """
        Initialize the miner with a predictive model.
        
        Args:
            config: Bittensor configuration (will use defaults)
            model: An instance of a PredictionModel. If None, uses SimpleAPIModel as default.
        
        Example:
            # Use default simple model
            miner = Miner()
            
            # Use your custom model
            from my_models import MyCustomModel
            my_model = MyCustomModel()
            miner = Miner(model=my_model)
        """
        super(Miner, self).__init__(config=config)
        
        # ============================================================
        # STEP 1: SETUP YOUR MODEL
        # ============================================================
        # Replace SimpleAPIModel() with your own model instance.
        # Your model must implement the PredictionModel interface.
        #
        # Example:
        #   from my_models import MyCustomModel
        #   self.model = MyCustomModel()
        #
        # ============================================================
        
        if model is None:
            bt.logging.info("No model provided, using SimpleAPIModel as default")
            self.model = SimpleAPIModel()
        else:
            self.model = model
        
        # Initialize the model (load weights, connect to APIs, etc.)
        if not self.model.initialize():
            bt.logging.warning(
                "Model initialization returned False. "
                "Miner will continue but predictions may fail."
            )
        else:
            bt.logging.success("Model initialized successfully")

    async def forward(self, synapse: bittbridge.protocol.Challenge) -> bittbridge.protocol.Challenge:
        """
        Responds to the Challenge synapse from the validator.
        
        This method:
        1. Extracts the timestamp from the synapse
        2. Calls your model's predict() method
        3. Attaches the prediction and interval to the synapse
        4. Returns the synapse to the validator
        
        Don't need to modify this method - it delegates to your model.
        The validator will score your predictions based on insentive mechanism.
        
        Args:
            synapse: Challenge synapse containing the timestamp to predict from
        
        Returns:
            Challenge synapse with prediction and interval filled in
        """
        # Extract timestamp from the challenge
        timestamp = synapse.timestamp
        
        bt.logging.debug(f"Received challenge for timestamp: {timestamp}")
        
        # ============================================================
        # STEP 2: GET PREDICTION FROM YOUR MODEL
        # ============================================================
        # This is where your model's predict() method is called.
        # Your model should return:
        #   - prediction: float (the predicted USDT/CNY price)
        #   - interval: [lower, upper] (90% confidence interval)
        #
        # If your model fails, it should return (None, None)
        # ============================================================
        
        try:
            prediction, interval = self.model.predict(timestamp)
            
            # Handle model failure
            if prediction is None:
                bt.logging.warning(
                    f"Model returned None prediction for timestamp {timestamp}. "
                    "Validator will ignore this response."
                )
                return synapse  # prediction and interval remain None
            
            # Attach prediction to synapse
            synapse.prediction = prediction
            
            # Attach interval to synapse (convert to list if needed)
            if interval is not None:
                synapse.interval = list(interval) if not isinstance(interval, list) else interval
            else:
                bt.logging.warning(
                    f"Model returned None interval for timestamp {timestamp}. "
                    "Only point prediction will be scored."
                )
            
            # Log successful prediction
            bt.logging.success(
                f"Prediction for {timestamp}: {prediction}, "
                f"Interval: {synapse.interval}"
            )
            
        except Exception as e:
            # Handle unexpected errors gracefully
            bt.logging.error(
                f"Error in model.predict() for timestamp {timestamp}: {e}"
            )
            # Return synapse with None values - validator will ignore
            return synapse
        
        return synapse

    async def blacklist(self, synapse: bittbridge.protocol.Challenge) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted.
        
        This method filters requests before they are processed. You can customize
        this to implement your own security policies.
        
        Current implementation:
        - Rejects requests without hotkeys
        - Optionally rejects non-registered entities
        - Optionally rejects non-validators
        
        Args:
            synapse: Challenge synapse (headers only, data not deserialized yet)
        
        Returns:
            Tuple[bool, str]: (should_blacklist, reason)
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # ============================================================
        # STEP 3: CUSTOMIZE BLACKLIST LOGIC (OPTIONAL)
        # ============================================================
        # You can add custom blacklist logic here, such as:
        # - Rate limiting
        # - IP-based filtering
        # - Reputation-based filtering
        # ============================================================
        
        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            # Hotkey not found in metagraph
            if not self.config.blacklist.allow_non_registered:
                bt.logging.trace(
                    f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"
            else:
                return False, "Non-registered hotkey allowed"

        # Check if validator permit is required
        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bittbridge.protocol.Challenge) -> float:
        """
        Determines the priority of incoming requests.
        
        Higher priority requests are processed first. This implementation
        prioritizes based on the validator's stake in the metagraph.
        
        Args:
            synapse: Challenge synapse
        
        Returns:
            float: Priority score (higher = process first)
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0
        try:
            caller_uid = self.metagraph.hotkeys.index(
                synapse.dendrite.hotkey
            )
            priority = float(self.metagraph.S[caller_uid])
        except (ValueError, IndexError):
            # Hotkey not found or invalid UID
            priority = 0.0
        
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority
    


# ============================================================
# MAIN ENTRY POINT
# ============================================================
# This is where the miner starts.
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # STEP 5: INSTANTIATE YOUR MODEL
    # ============================================================
    # Replace SimpleAPIModel() with your own model.
    #
    # Example:
    #   from my_models import MyCustomModel
    #   model = MyCustomModel()
    #   miner = Miner(model=model)
    # ============================================================
    
    # Use the simple example model (or replace with your own)
    model = SimpleAPIModel()
    
    # Create and run the miner
    with Miner(model=model) as miner:
        bt.logging.info("Miner started. Waiting for challenges from validators...")
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
