'''
Created on 26.01.2020

@author: godly_000
'''
import cProfile, pstats

pr = cProfile.Profile()
        
def startProfiling():   
    pr.enable()
    
def stopProfiling(numResults=10):
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(numResults)
    pstats.Stats(pr).sort_stats("time").print_stats(numResults)