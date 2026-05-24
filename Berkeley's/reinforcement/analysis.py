# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).

      Low discount → close +1 worth more than distant +10.
      Noise=0 → deterministic, can walk right along the cliff edge safely.
      The "risk" is proximity — path passes adjacent to cliff cells.
    """
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).

      Low discount → prefer close +1.
      Noise=0.2 → cliff IS actually dangerous with random actions.
      Zero living reward → extra safe-path steps cost nothing.
      Agent takes the longer but cliff-free route.
    """
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).

      High discount → distant +10 worth the walk.
      Noise=0 → can skirt the cliff edge safely (the "risk" is proximity).
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).

      High discount → distant +10 is worth it.
      Noise=0.2 → cliff is realistically dangerous.
      Agent takes the safe, longer path to the distant exit.
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).

      Positive living reward → every step earns points.
      Agent prefers to wander forever collecting living reward.
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
