"""
Module with invoke tasks
"""

import invoke
import src.invoke.tests
#import net.invoke.train


# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(src.invoke.tests)
#ns.add_collection(net.invoke.train)