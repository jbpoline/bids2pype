info = [
    Bunch(  conditions=['cond1'], 
            onsets=[[2, 50, 100, 180]],                      
            durations=[[1]]
            ),

    Bunch(  conditions=['cond1'], 
            onsets=[[30, 40, 100, 150]],                       
            durations=[[1]]
            )
]


Bunch(  conditions=['cond1', 'cond2'],                       
        onsets=[[2, 50],[100, 180]], 
        durations=[[0],[0]],                       
        pmod=[ Bunch( name=['amp'], 
                      poly=[2], 
                      param=[[1, 2]]),                      
               None ]
        ),

Bunch(  conditions=['cond1', 'cond2'],                       
        onsets=[[20, 120],[80, 160]], 
        durations=[[0],[0]],                       
        pmod=[  Bunch(  name=['amp'], 
                        poly=[2], 
                        param=[[1, 2]]
                        ),                       
                None ]
        )


