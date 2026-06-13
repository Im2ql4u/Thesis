first we must redo the aproach, we can first itroduce some terminology, the overall scope of paper A, and paper B. the MBSE problem (FCI, NQS, VMC, HF, CCSD and DMC), then our ansatz, then CTNN architecture. 

We introduce first paper A,
on Q1, we mention no unified theory, double decent has been theorized/observed somewhat, NTK is an intresting theory, you must write a proper explaination for me on how this works, explain the norm stuff for me as well, but don't include it.

On expressitivity, include the example from the paper on a neural network calculating the norm between two objects, scaling exponentially in d for 1 layer, polynomically for multi-layers, we also mention the genral universal aproximation theory, but also discuss what in the paper they say width is important for.

on CoD, we should go into the instrinsic dimenstionality, mention that FCI and HF takes effectively i.e 40 d problem and scales it to an infinitely large matrix to solve, then explain how CCSD does not do this, but sacrifices expressitivity for scaling, and we know where it fails miserably (wigner regimes), the idea of neural networks is to scale well with d, while being expressive at the same time, this is in my eyes the purpose of working on ML. and the paper discusses barron functions, you must explain that proof to me in much more depth, and clearly, and then explain how that does not apply for us since we have coloumb stuff, but how how we can try and condition the problem to become more barron-like, which we go into more depth on later.explain what s is in that definition of the curse

on training, explain natural gradients vs ADAM, and ADAM vs SGD, how batching helps with the slight noise, show how NTK is linear and what that implies, explain that to me, explain what that spectral bias is to me, dont include it in the presentation

Much of paper B can be mentioned shortly, the  Q1 is just saying whether neural network have expressitivity, which we know from the universal approximation theory, and its later extensions to relu like networks, they intrduce a sobolev norm, and show the same thing again for pinns, arguing more about how linear models are restructed, this is un-interesting. we can mention this, but im going to compare many body techniques and expressitivityt more.

Q2 can be mentioned very shortly, it assumes a low generalization loss, is then the training loss small, which for us obviously is the case, if we have a true GS, both variance will be low, and the system will be a perfect eigenstate. we show this with some formulas.

 then Q3 adresses whether a small loss in training implies a small total actual real pde loss. for MBSE we always evaluate with a VMC calcualtion, hence it is no mystery for us, and their main points are that we must sample effectively, we should therefore introduce how we sample

 Then we move on to 4, which mentions how hard it is to train. you must write a good summary for me what they come to here, and how they aproach this results, what exactly is NTK, LL*, and T*T and kappa, lambda and such. link these up to coloumb repulsion and our laplacian, and how they bring errors with them. discuss also how our backflow specifically is a problem here, bring up our catch-22 results and how nothing helped except removing the laplacian from the backflow entirely, but keeping its term as a loss term, this shows specifically how conditioning helps. also bring up why since we have an analytic cusp, this likely works well, and why that term is important. 

But then we focus on presenting our results, 
then we show both our SR+VMC results for bf and ctnn, and also our pinn results for the ctnn structure, we must discuss what tricks we did in the pinn trinning that helped, where SR breaks down and why, and also why ctnn helps so much.

then we discuss our ansatz and the inputs we have, why did we do the inputs we did, we analyzed which were used, what does this imply, are the regime shifts in the inputs used actually because og gradient stabilities, or are they regime-shift focus? have we done any work rying to see if gradient-instabilities with the inputs show in what the network prioritizes. we link this with the theory developed in paper B on training stability.

then we show our intrinsic dimensionality results, and include the dimensionality results we had for ctnn as well if we had any for the messaging and stuff, what could we measure dimension of, could we run something now? it is harder to run pca the same way because it produces a 40d output per particle per dimension thing, so it will vary across all those dimensions if we measure that way. we discuss here how the neural nets dont scale awfully with d, but finds the intrisic dimensions of the problems, what are these? how do they scale for 2-6-12-20 electrions? what do we know here? this is an intersting discussion.

you must explain to me more clearly what the three body thing is, and what it implies.

rmemeber, keep as little on the presentaiton as possible, every term, every formula will be qquesitoned and assumed i know in and out, therefore we shouldnt mention a bullion things, but mostly just what is directly relevant to us. and also, keep sitations mostly to the papers themselves, so we dont do something wrong there. and discuss to me how NTK is linked with SR

 