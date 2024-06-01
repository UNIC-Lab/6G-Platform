# -*- coding: utf-8 -*-

import os
import time

import gradio as gr
from transformers import pipeline

from kpi_extract2 import KpiExtractor
from stitch_system import StitchSystem


def load_resnet18():
    return pipeline("image-classification", model="microsoft/resnet-18", device="cuda:0")

def inference(delay, accuracy, image_data):
    delay = float(delay)
    accuracy = float(accuracy)
    start_time = time.time()
    time_cost = None
    classifier = load_resnet18()
    outputs = classifier(image_data)

    time_step = delay / len(image_data) / 1000
    
    results = []
    for i, sample_output in enumerate(outputs):
        results.append((image_data[i], f"{sample_output[0]["label"]}: {sample_output[0]["score"]:.2f}"))
        tmp_delay = (i + 1)*time_step - (time.time() - start_time)
        if tmp_delay > 0:
            time.sleep(tmp_delay)
        yield results, f"{(time.time() - start_time)*1000} ms"

    time_cost = f"{(time.time() - start_time)*1000} ms"
    yield results, time_cost

with gr.Blocks() as demo:
    net_plot = gr.Plot(label="Network Architecture")
    input_header = gr.Markdown("# Input")
    prompt = gr.Textbox(label="User Prompt")
    images_data = gr.File(file_count="multiple")
    submit = gr.Button("Submit")
    output_header = gr.Markdown("# Output")
    kpi_header = gr.Markdown("## Resolved KPIs")
    with gr.Row():
        kpi_acc = gr.Text(label="Accuracy (%)")
        kpi_total_time = gr.Text(label="Total Time Cost (ms)")
        kpi_comm_time = gr.Text(label="Communication Time Cost (ms)")
        kpi_comp_time = gr.Text(label="Computation Time Cost (ms)")
        kpi_throughput = gr.Text(label="Throughput (Mbps)")
    with gr.Row():
        real_time_cost = gr.Text(label="Real Time Cost")
    inference_header = gr.Markdown("## Inference Results")
    gallery = gr.Gallery(
        label="Results",
        columns=[9],
        rows=[1],
        object_fit="contain",
        height="auto"
    )

    prompt_examples = gr.Examples(
        examples=[
            ["Completing this task in 10s is ok, but I require greater than 80% accuracy."],
            ["I want to finish this task in 3 seconds with accuracy of 70%."],
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

    def resolve_kpi(prompt, images_data):
        extractor = KpiExtractor()
        the_demands = extractor.kpi_extract(prompt)
        system_simulator = StitchSystem()
        print(the_demands["accuracy"], the_demands["delay"])
        the_kpis = system_simulator.cal_delay(the_demands["accuracy"], the_demands["delay"], 3*224*224*8/1024/1024, len(images_data))
        # 模型1索引，模型2索引，缝合位置，模型1部署网络层，模型2部署网络层，最小总时间，最小传输时间, 最小推理时间，可容忍带宽
        print(the_kpis)
        real_model_acc = the_kpis[0]
        pre_model_index = the_kpis[1]
        post_model_index = the_kpis[2]
        stitch_position = the_kpis[3]
        pre_model_location = the_kpis[4]
        post_model_location = the_kpis[5]
        total_time = the_kpis[6]
        comm_time = the_kpis[7]
        comp_time = the_kpis[8]
        throughput = the_kpis[9]
        return {
            kpi_acc: real_model_acc,
            kpi_total_time: total_time,
            kpi_comm_time: comm_time,
            kpi_comp_time: comp_time,
            kpi_throughput: throughput,
        }

    submit.click(
        fn=resolve_kpi,
        inputs=[prompt, images_data],
        outputs=[kpi_acc, kpi_total_time, kpi_comm_time, kpi_comp_time, kpi_throughput]
    ).then(
        fn=inference,
        inputs=[kpi_total_time, kpi_acc, images_data],
        outputs=[gallery, real_time_cost]
    )

demo.launch(share=True)

