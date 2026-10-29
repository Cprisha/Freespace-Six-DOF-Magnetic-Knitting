#Magnetic Knitting 
Many believe the art of knitting is old-school and disappearing. Yet, everyone wants knitted jumpers and clothing that scream "MY-GRANDMA-KNITTED-IT", even if she didn't. But, the art of knitting is known to a few and considering how busy the world is, even fewer are able to add this activity into the schedule. So, what if, the desired hand woven effect can be achieved? Not by making 24 hours 25, rather by using magnetic actuation to knit.  

## Inspiration:
I grew up reading Harry Potter and when I saw the knitting charm in BURROW (Weasley's House, Harry Potter and the Chamber of Secrets) I was obsessed. As I grew up, I realised- though magic is not real, physics is basically applied magic- with logic being the wand and spells still being discovered.

## Suggestion:
Using electromagnetic solenoid array in order to build a self knitting machine using magnetic actuation.

## Approach:
1. Base
- A suitable base on top of a chair/sofa like frame with two halves separated and bordered with a strong insulating material.
- Each half contains an array of electromagnets.
- A knitting needle sits over each array.
2. Knitting Needle
- Needles are made of ferromagnetic material or aluminium alloy for predictable magnetic response.
- They have an embedded magnetic strip running internally, not continuous, to help stabilize rotation and control movement.
- The ends (not the pointed strips) are flat to give magnetic field a better surface for torque transfer.
- The needle has a thin mu-metal coating for passive shielding to prevent cross interference of magnetic fields.
3. Electromagnet array
- The base uses a dense grid of miniature solenoids, each individually addressable, to reduce unwanted interactions.
- The circuit incorporates resistors and capacitors for minimizing heat loss and increasing stability.
- Solenoids are driven by pulse-width-modulation (PWM) to ensure smooth movements and proper timings.
- The power supply is fully external.

## Mechanism:
1. The system energizes the array of solenoids in a carefully preprogrammed pattern using PWM, which helps sequencing attractive-repulsive fields as required. As current passes through them, a magnetic field is generated (Maxwell's Laws). Using this field, magnetic actuation is achieved.
2. The first set of instruction from the PWM ensure the correct positioning of the needles above the  array. The initial magnetic field levitates the magnetic needle to the desired position from where the real set of knitting actions start.
3. Here the solenoids basically work similar to an AC but not exactly it. Controlled vibration are generated in the needle due to timed oscillation of the solenoids.
4. By controlling which solenoid gets current when and by how much, we can move, lift and rotate the knitting needles as needed. As we change the direction of current in a solenoid, the direction of the magnetic field induced also changes.
5. As eddy current is induced in the needles, which causes the rotation to become smoother.
6. As we decrease and increase the current passing through the solenoids (transition being smooth), we can fluctuate the magnetic field produced and hence change the positioning of the needles.
7. By proper co-ordination of the knitting instruction cycle, we can replicate the desired knitting pattern.
8. Adjacent solenoids never activate in conflicting patterns to prevent cross-interference of magnetic fields.
9. Cross interference of opposite magnetic fields can be calculated and the pattern is set keeping that in mind.

## Major Challenges:
1. Magnetic Localization
If A, B and C are neighbouring solenoids, then, the magnetic field of A should not interfere with the magnetic field of adjacent solenoids B and C.
Solution: Using PWM we can ensure neighbouring solenoids do not have opposite direction of field at any point.
2. Thermal Management
Solenoids heat up on constant usage, hence, decreasing the life of the machine and it might harm the wool.
Solution: Using PWM to switch solenoids continuously to prevent any major heat loss. Increasing the gap between adjacent solenoids and using solenoids of laminated cores made of silicon (or similar metal alloy) sheets.

## Questions and Answers:
1. Won't the gravity restrict the needle from levitating?
Ans. The effect of gravity on the needle does not hinder its functioning. The force due to the induced magnetic field can be calibrated so that its vertical component cancels out the weight (mg) of the needles.
2. What about the knitted fabric?
Ans. The knitted fabric gets collected in a separate wooden structure below the work area.

## Warning:
1. Projectile Effect/Missile Effect: The system might collapse if a ferro-magnetic substance enters the field.
2. Strong Magnetic Field: The magnetic field might be too strong and can harm the human body over a prolonged period of time.

## Considerations:
1. There should be an active noise cancellation system as rapid firing of solenoids might be loud.
2. Ensure a backup power supply, sudden disruption can cause the solenoids to fuse.
3. Calculate the magnetic flux and set the PWM in such a manner that it can be avoided.

## Future Upgrades: 
1. Using AI and computer vision to add detailed knots and patterns, and for real-time error detection.
2. Creating a multi needle system for full time knitting machine.

## Similar Project:
Magnetic array knitting needle driving system design (Authors: Chengjun Zhang, Dongdong Li, Xiaoyan Zuo, Xiangyang Zhou): A similar concept (I found much later while checking my idea for unintentional plagiarism) but not exactly the same.
