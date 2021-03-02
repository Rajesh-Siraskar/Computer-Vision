# xenon
### Computer Vision and Machine Learning
------------------------------------------------------------------------------------------------------

### Update: 31-Dec-2018 ###
-------------------

### Idea: Perspective map method ###
 1. Start with horizons at 30% from top, then 40% and then 50%
 2. For each horizon, use box sizes from 0.25, 0.5, 1.0, 1.5 and 2.0 times trained BB sizes
 3. Scan left to right with 0.2 of above box width
 4. Catching kids: Kids will be a smaller box but on higher horizon. So a kid on 305 hoprizon will hopefully be identified by the smaller box at 50% horizon


<B>Files</B>:
- <B>xenon_CodeLab</B>: Experimental code - To develop utility code and code tit-bits to be later used for building the assignment

<B>Log</B>:
- 07-Dec-2018: Begin work on assignment
- 08-Dec-2018: What does HoG do? 


08-Dec-2018: What does HoG do?
1. Can we take a simple 2D figure like a house shape and plot HoGs to see what it looks like?
2. How does it look for a rect.? A triangle? and then the house? A circle?
3. A steep line vs a gently sloped line

08-Dec-2018: Understanding what gradients mean for images
- (Klette, 2014: 11): See 'Detecting Step-Edges by First- or Second-Order Derivatives'
- 1st order and 2nd order derivatives provide edge detection
- The 'curve' on which to do derivatives are intensities? Along X or Y axes
- Check using simple high contrast images developed in MS Paint
- See if we can plot the intensities along sections of a square and triangle and see expected itensity peaks

**REFERENCES**:
1. Pg.11-13: Detecting Step-Edges by First- or Second-Order Derivatives
2. Pg.62-63: Basic edge detection
3. Pg.382-384: HoG algorithm
