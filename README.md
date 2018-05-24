# TestProject
My Sandbox Project for playing around with Python and github.

My initial goal is to develop my own jpeg compression and decompression functions.  Since I'm doing this to learn, I'm not using a bunch of the built-in numpy functions that would make it easier.  I'm writing my own unittested DCT functions instead.  I also plan to write some jpeg file IO functions.

In addition, I plan to play around with setting up Jenkins continuous integration.  Currently I have CI setup to test all github branches and pull-requests, however the server is running on my local PC meaning the test reports are not available to an external user.  In the future I'd like to move Jenkins to an EC2 instance on AWS and make it accessible to the project contributors.
