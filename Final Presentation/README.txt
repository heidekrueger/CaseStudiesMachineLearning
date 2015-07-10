pgfopts - LaTeX package options with pgfkeys
============================================

Using key-value options for packages and macros is a good way of
handling large numbers of options with a clean interface. The
`pgfkeys` package provides a very well designed system for
defining and using keys, but does not make this available for
handling LaTeX class and package options. The `pgfopts` package
adds this ability to `pgfkeys`, in the same way that `kvoptions`
extends the `keyval` package.

Installation
------------

The package is supplied in `.dtx` format and as a pre-extracted
zip file, `pgfopts.tds.zip`. The later is most convenient for
most users: simply unzip this in your local `texmf` directory.
If you want to unpack the `.dtx` yourself, running `tex
pgfopts.dtx` will extract the package whereas 'latex pgfopts.dtx
will extract it and also typeset the documentation.

Typesetting the documentation requires a number of packages in
addition to those needed to use the package. This is mainly
because of the number of demonstration items included in the
text. To compile the documentation without error, you will
need the packages:
 - `csquotes`
 - `helvet`
 - `hypdoc`
 - `listings`
 - `lmodern`
 - `mathpazo`
 - `microtype`
