from matplotlib import pyplot as plt
import matplotlib

marker=ur'$\u263A$'

plt.plot(0, 1, "g", marker=marker, markersize=20)
plt.plot(2, 1, "g", marker=ur'$\u263C$', markersize=20)

plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.xlim( -1, 5 )
plt.ylim( 0, 2 )

plt.show()
