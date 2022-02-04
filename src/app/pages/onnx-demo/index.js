import React, {memo, useEffect, useState} from 'react';
const ort = require('onnxruntime-web');
function OnnxDemo(props) {
    const [embedding, setEmbedding] = useState([]);
    const [inputIds, setInputIds] = useState([1, 5, 5, 6, 5, 6, 0, 7, 27, 5, 6, 13, 5, 6, 0, 7, 0, 5, 0, 5, 11, 5, 0, 5, 11, 5, 0, 0, 6, 7, 8, 7, 0, 5, 7, 6, 6, 13, 7, 27, 5, 5, 0, 5, 11, 5, 12, 7, 0, 11, 2]);
    const [attentionMask, setAttentionMask] = useState([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    useEffect(() => {
        const inference = async () => {
            try {
                // Load Onnx Models and create inference session
                const session = await ort.InferenceSession.create('/onnx_models/model.onnx');

                // prepare inputs. a tensor need its corresponding TypedArray as data
                const dataA = BigInt64Array.from(inputIds, x => BigInt(x));
                const dataB = BigInt64Array.from(attentionMask, x => BigInt(x));
                const input_ids = new ort.Tensor('int64', dataA, [1, inputIds.length]);
                const attention_mask = new ort.Tensor('int64', dataB, [1, attentionMask.length]);

                // prepare feeds. use model input names as keys.
                const feeds = {input_ids: input_ids, attention_mask: attention_mask};

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                const embeddedInputIds = results['854'].data;
                setEmbedding(embeddedInputIds)

            } catch (e) {
               console.log(`Something Went Wrong! ${e}`)
            }
        }
        inference().then(console.log('Finished Embedding!'))
    }, [inputIds, attentionMask]);
    console.log(embedding)
    return <div><p>Bonjour</p></div>
}

export default memo(OnnxDemo);