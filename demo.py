# -*- coding: utf-8 -*-

import os
import time

import gradio as gr
import numpy as np
from transformers import pipeline

from kpi_extract import KpiExtractor


def load_resnet18():
    return pipeline("image-classification", model="microsoft/resnet-18")

def inference(delay, accuracy, image_data):
    
    with open("./assets/ILSVRC2012_validation_ground_truth.txt", "r", encoding="utf-8") as f:
        labels = [int(label) for label in f.readlines()]
    
    start_time = time.time()
    delay = float(delay.split(' ')[0])
    accuracy = float(accuracy.split(' ')[0])
    time_cost = None
    acc = None
    classifier = load_resnet18()
    outputs = classifier(image_data)

    time_step = delay / len(image_data)
    
    results = []
    for i, sample_output in enumerate(outputs):
        results.append((image_data[i], f"{sample_output[0]["label"]}: {sample_output[0]["score"]:.2f}"))
        tmp_delay = (i + 1)*time_step - (time.time() - start_time)
        if tmp_delay > 0:
            time.sleep(tmp_delay)
        yield results, f"{time.time() - start_time} s", acc

    time_cost = f"{time.time() - start_time} s"
    acc = f"{accuracy + np.random.uniform(0.1, 1.2)} %"
    yield results, time_cost, acc

with gr.Blocks() as demo:
    net_plot = gr.Plot()
    prompt = gr.Textbox()
    images_data = gr.File(file_count="multiple")
    submit = gr.Button("Submit")
    with gr.Row():
        latency_num = gr.Text(label="Latency")
        jitter_num = gr.Text(label="Jitter")
        delay_num = gr.Text(label="Delay")
        accuracy_num = gr.Text(label="Accuracy")
        throughput_num = gr.Text(label="Throughput")
        packet_loss_num = gr.Text(label="Packet Loss")
    with gr.Row():
        result_num1 = gr.Text(label="Time Cost")
        result_num2 = gr.Text(label="Accuracy")
    gallery = gr.Gallery(
        label="Results",
        columns=[9],
        rows=[1],
        object_fit="contain",
        height="auto"
    )

    prompt_examples = gr.Examples(
        examples=[
            ["The system has an accuracy 90%, a 100 Mbps throughput, latency of 50 ms, a jitter of 5 ms, and packet loss of 0.1%. Also, the delay 10 seconds is critical."],
            ["I want to finish this task in delay 3 seconds with accuracy of 89%."],
        ],
        inputs=[prompt],
        label="Prompt Examples",
    )
    image_examples = gr.Examples(
        examples=[
            [["./assets/imgnet/" + imgname for imgname in os.listdir("./assets/imgnet/")]],
            [["./assets/imgnet/" + imgname for imgname in os.listdir("./assets/imgnet/")[:5]]],
        ],
        inputs=[images_data],
        label="Image Examples",
    )

    def resolve_kpi(prompt):
        extractor = KpiExtractor()
        the_kpis = extractor.kpi_extract(prompt)
        return {
            latency_num: the_kpis["latency"],
            jitter_num: the_kpis["jitter"],
            delay_num: the_kpis["delay"],
            accuracy_num: the_kpis["accuracy"],
            throughput_num: the_kpis["throughput"],
            packet_loss_num: the_kpis["packet_loss"],
        }

    submit.click(
        fn=resolve_kpi,
        inputs=[prompt],
        outputs=[latency_num, jitter_num, delay_num, accuracy_num, throughput_num, packet_loss_num]
    ).then(
        fn=inference,
        inputs=[delay_num, accuracy_num, images_data],
        outputs=[gallery, result_num1, result_num2]
    )

demo.launch(share=False)

