import streamlit as st
import torch
from PIL import Image
from io import BytesIO
import os
from transformer_net import SimpleTransformerNet
from transformer_netv2 import TransformerNet as TransformerNet
from torchvision.transforms import functional as TF
from utils import tensor_to_image
import time

# ------ PAGE CONFIG & STYLING ------
st.set_page_config(
    page_title="Neural Style Transfer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced visuals
st.markdown("""
<style>
body {background-color: #f5f5f5;}
.sidebar .sidebar-content {background-color: #ffffff;}
.stApp {font-family: 'Arial', sans-serif;}
h1, h2, h3 {color: #333333;}
</style>
""", unsafe_allow_html=True)

# ------ HEADER ------
st.markdown("# 🎨 Neural Style Transfer Studio")
st.markdown(
    "Welcome to the **Neural Style Transfer Studio**, where you can apply different artistic styles to your photos in real-time. "
    "Choose a pre-trained model, adjust parameters, and transform your images into artwork!"
)

# ------ MODEL LOADING ------
# Find saved models
base_dir = os.path.dirname(__file__)
saved_v1 = os.path.join(base_dir, "saved_models")
saved_v2 = os.path.join(base_dir, "saved_modelsv2")
models = []
if os.path.isdir(saved_v1):
    models += [(f, 'v1') for f in sorted(os.listdir(saved_v1)) if f.endswith('.pth')]
if os.path.isdir(saved_v2):
    models += [(f, 'v2') for f in sorted(os.listdir(saved_v2)) if f.endswith('.pth')]

if not models:
    st.error("No style models found in saved_models or saved_modelsv2.")
    st.stop()

# Sidebar: Model selection
st.sidebar.header("🎨 Style Model & Settings")
model_name = st.sidebar.selectbox("Select Style Model:", [m[0] for m in models])
version = next(v for n, v in models if n == model_name)
model_path = (saved_v1 if version == 'v1' else saved_v2) + '/' + model_name
alpha = st.sidebar.slider(
    "Style Strength (alpha)", 0.0, 1.0, 1.0, step=0.05,
    disabled=(version != 'v1'),
    help="Only applies to v1 models"
)

# Sidebar: Content upload
st.sidebar.markdown("---")
content_file = st.sidebar.file_uploader("Upload Your Photo", type=["jpg","jpeg","png"])
st.sidebar.markdown(
    "Use this app to experiment with different artistic styles.\n"
    "Adjust the strength slider for subtle to strong stylization."
)

# ------ MODEL LOADING FUNCTION ------
@st.cache_resource
def load_model(path, ver):
    net = SimpleTransformerNet() if ver == 'v1' else TransformerNet()
    state = torch.load(path, map_location='cpu')
    # strip old InstanceNorm stats
    for k in list(state.keys()):
        if k.endswith(('running_mean', 'running_var')):
            del state[k]
    net.load_state_dict(state, strict=False)
    net.eval()
    return net

# ------ MAIN LAYOUT ------
col1, col2 = st.columns((1,1))

# ------ EXAMPLE BEFORE & AFTER ------
with st.expander("🔍 See a Quick Before & After Demo", expanded=True):
    st.markdown(
        "Here’s how the style transfer works on a sample image. "
        "Upload your own photo below to apply the selected artistic style."
    )
    ex_col1, spacer, ex_col2 = st.columns([3,1,3])
    try:
        ex_input = Image.open(os.path.join(base_dir, 'Before & After', 'COCO_train2014_000000000036.jpg')).convert('RGB')
        ex_output = Image.open(os.path.join(base_dir, 'Before & After', 'stylized (1).png')).convert('RGB')
    except Exception:
        ex_input = ex_output = None
    ex_col1.subheader("Before")
    if ex_input:
        ex_col1.image(ex_input, width=200)
    else:
        ex_col1.write("_Example content image not found._")
    ex_col2.subheader("After")
    if ex_output:
        ex_col2.image(ex_output, width=200)
    else:
        ex_col2.write("_Example stylized image not found._")

# ------ MAIN CONTENT ------


if content_file:
    # Display original image
    img = Image.open(content_file).convert('RGB')
    col1.subheader("📷 Original")
    col1.image(img, use_column_width=True)

    # Prepare image
    w, h = img.size
    w_crop, h_crop = w - w % 4, h - h % 4
    img = img.crop((0, 0, w_crop, h_crop))

    # Preprocess
    tensor = TF.to_tensor(img).unsqueeze(0)
    if version == 'v2':
        tensor *= 255.0
    tensor = tensor.to('cpu')

    # Load model
    with st.spinner("Loading style model..."):
        model = load_model(model_path, version)
        model.to('cpu')

    # Stylize
    start = time.time()
    with torch.no_grad():
        output_tensor = model(tensor, alpha=alpha) if version == 'v1' else model(tensor)
    stime = time.time() - start

    output_tensor = output_tensor.squeeze(0)

    # Postprocess
    if version == 'v2':
        arr = output_tensor.clamp(0,255).numpy().transpose(1,2,0).astype('uint8')
        output_img = Image.fromarray(arr)
    else:
        output_img = tensor_to_image(output_tensor)

    # Display stylized image
    col2.subheader(f"🖼 Stylized (v{version[-1]}) — {stime:.2f}s")
    col2.image(output_img, use_column_width=True)

    # Download option
    buf = BytesIO()
    output_img.save(buf, format='PNG')
    col2.download_button(
        label="Download Stylized Image",
        data=buf.getvalue(),
        file_name="stylized.png",
        mime="image/png"
    )
else:
    st.info("📤 Upload an image to see the magic!")

# ------ FOOTER ------
st.markdown("---")
st.markdown(
    "Built with ❤️ using **PyTorch** and **Streamlit**. "
    "Explore more on [GitHub](https://github.com/your-repo)"
)
