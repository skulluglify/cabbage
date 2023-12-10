====
Python Projekt (Dataset)
====

- label.txt
.. code-block:: text

    0.0 0.5 0.5 1.0 1.0
    1.0 0.5 0.5 0.75 0.75
..

- annotation.xml (coming soon)
.. code-block:: xml

    <images>
        <image src="../butterfly.png"/>
            <annotations>
                <annotation id="1" xtl="0.0" ytl="0.0" xbr="0.0" ybr="0.0">
                    <datapoints xtl="0.0" ytl="0.0" xbr="0.0" ybr="0.0">
                        <datapoint vw="640"/>
                        <datapoint vh="640"/>
                    </datapoints>
                </annotation>
            </annotations>
        </image>
    </images>
..

- dataset.py
.. code-block:: python

    dataset_projekt = ProjektDataset("datasets/projekt")
    dataset_projekt.thread(jobs=12)

    # Preload Images First (Optional)
    dataset_projekt.preload(step="train")

    for result in iter(dataset_projekt):
        ...
..