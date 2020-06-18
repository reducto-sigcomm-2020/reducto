# Reducto: On-Camera Frame Filtering for Resource-Efficient Real-Time Video Analytics

![logo](logo.png)

Real-time queries of live video can guide numerous applications to ﬁnd useful
information at a timely fashion. To extract such information from videos,
analytics pipelines rely on deep neural networks for tasks such as object
detection. Unfortunately, such DNNs impose high resource demands on backend
servers, which often prevent analytics pipelines from delivering low-latency
responses for real-time queries. Recent techniques reduce backend computation by
employing edge servers to filter out irrelevant frames using lightweight NN
models. However, edge servers pose numerous challenges with respect to
deployment costs and privacy, and they ignore potential network bottlenecks at
cameras.

This paper presents Reducto, a camera-server system that provides low enough
latency for answering real-time video queries. The key idea behind Reducto is
filtering out unnecessary frames *directly at the data source* (i.e., cameras)
to achieve additional efficiency. However, commodity cameras often lack the
resources (e.g., GPUs) required to run even lightweight NN models. Our key
observation is that techniques based on *differencing features* are very cheap
to run and can effectively track *frame differences*, which are well correlated
with changes in query results. Based on this observation, we built Reducto that
can effectively filter out frames at the camera based on the best feature
selected by the server for each query. Due to use of cheap techniques, the
camera can constantly check the validity of the feature and once it grows out of
date the server quickly selects a new one. As such, Reducto ensures that the
filtering technique adapts to the inherent dynamism of the video and meets the
desired accuracy at a ﬁne granularity. Comparisons between Reducto and existing
systems demonstrate very promising results.
