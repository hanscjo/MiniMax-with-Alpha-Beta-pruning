class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        pacman_actions = gameState.getLegalActions(0)
        max_value = float('-inf') # Instantiated as -inf so that some action will always be chosen
        decision = 0

        for action in pacman_actions:  # Minimax between all possible decisions
            action_value = self.Min_Value(gameState.generateSuccessor(0, action), 1, 0)  # Get the value of a given action in relation to another agent(player)

            if action_value > max_value:  # Maximizing for the root node
                max_value = action_value
                decision = action  # Selects the final minimaxed value

        return decision


    def Max_Value(self, gameState, depth):

        if (depth == self.depth) or (len(gameState.getLegalActions(0)) == 0): # Avoiding redundant calls
            return self.evaluationFunction(gameState)

        return max([self.Min_Value(gameState.generateSuccessor(0, action), 1, depth) for action in gameState.getLegalActions(0)]) # Max-weighting


    def Min_Value(self, gameState, agent_index, depth):

        agent_actions = gameState.getLegalActions(agent_index)

        if len(agent_actions) == 0:  # The given agent has no available actions
            return self.evaluationFunction(gameState)

        if agent_index < gameState.getNumAgents() - 1: #  We're still not done, so we continue weighting nodes
            return min([self.Min_Value(gameState.generateSuccessor(agent_index, action), agent_index + 1, depth) for action in agent_actions]) # Min-weighting

        else:   # We've reached the final agent
            return min([self.Max_Value(gameState.generateSuccessor(agent_index, action), depth + 1) for action in agent_actions]) # Min-weighting

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha = float('-inf')  # Instantiated as -inf to always have some value saved for the Max-value
        beta = float('inf')    # --------------- +inf ------------------------- saved for the Min-value
        pacman_actions = gameState.getLegalActions(0)
        decision = 0

        for action in pacman_actions: # Minimax between all possible decisions
            action_value = self.Min_Value(gameState.generateSuccessor(0, action), 1, 0, alpha, beta) # Get the value of a given action in relation to another agent(player)

            if alpha < action_value:  # Maximizing for the root node
                alpha = action_value
                decision = action # Selectes the final minimaxed value

        return decision


    def Min_Value(self, gameState, agent_index, depth, alpha, beta):

        action_value = float('inf') # Instantiated as inf so that some action will always be chosen
        agent_actions = gameState.getLegalActions(agent_index)

        if len(agent_actions) == 0:  # There are no available actions
            return self.evaluationFunction(gameState)

        for action in agent_actions:
            if agent_index < gameState.getNumAgents() - 1: #  We're still not done, so we continue weighting nodes
                action_value = min(action_value, self.Min_Value(gameState.generateSuccessor(agent_index, action), agent_index + 1, depth, alpha, beta)) # Min-weighting

            else:  # We've reached the final agent
                action_value = min(action_value, self.Max_Value(gameState.generateSuccessor(agent_index, action), depth + 1, alpha,beta)) # Min-weighting

            if action_value < alpha: # Pruning
                return action_value

            beta = min(beta, action_value) # Update beta if necessary

        return action_value


    def Max_Value(self, gameState, depth, alpha, beta):

        action_value = float('-inf') # Instantiated as -inf so that some action will always be chosen
        pacman_actions = gameState.getLegalActions(0)

        if depth == self.depth or len(pacman_actions) == 0: # Avoiding redundant calls
            return self.evaluationFunction(gameState)


        for action in pacman_actions:
            action_value = max(action_value, self.Min_Value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta)) # Max-weighting

            if action_value > beta: # Pruning
                return action_value

            alpha = max(alpha, action_value) # Update alpha if necessary

        return action_value
