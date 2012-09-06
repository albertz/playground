# http://stackoverflow.com/questions/12292151/crash-in-class-getname-in-applicationopenuntitledfile

from AppKit import *

def setupWindow():
	w = NSWindow.alloc()
	w.initWithContentRect_styleMask_backing_defer_(
		((200.0, 200.0), (250.0, 100.0)),
		NSTitledWindowMask |
		NSClosableWindowMask |
		NSResizableWindowMask,
		NSBackingStoreBuffered, False)
	w.setTitle_("Hello world")

	w.display()
	w.orderFrontRegardless()
	w.makeMainWindow()

	app.delegate()._mainWindow = w

	return w

class PyAppDelegate(NSObject):

	def applicationOpenUntitledFile_(self, app):
		print "applicationOpenUntitledFile_", app
		print "delegate:", app.delegate()
		print self.__class__
		setupWindow()

app = NSApplication.sharedApplication()
appDelegate = PyAppDelegate.alloc().init()
app.setDelegate_(appDelegate)
app.finishLaunching()
app.run()
