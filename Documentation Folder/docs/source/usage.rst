Usage
=====

Installation
------------

The following are installation instructions for windows.

Before begining ensure that you have python installed with pip. The code is tested for python 3.8 so you may have to deprecative to this 
to ensure correct functioning.

To install and use the package, first download the zipped package from this link, and decompress the file in a folder of your choice.

Navigate to the decompressed folder and run the following command (with adminisitrator priviledges ):

.. code-block:: console

   (.venv) $ python setup.py install

This should then install abcpyscoringrules and all its required packages.
The package should then be accessable via the following command

.. code-block:: python

    import abcpyscoringrules 

and functions can be imported from the package like

.. code-block:: python

    from abcpyscoringrules.inferences import adSGLD, SGLD

as needed.

