import torch
import torch.nn as nn

# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# --- Hybrid CNN (same as in Colab) ---
class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNN, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x64
        )
        self.res1 = ResidualBlock(32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32x32
        )
        self.res2 = ResidualBlock(64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


# Minimal Streamlit UI so the app renders when run with `streamlit run app.py`.
# This intentionally keeps logic simple: shows that Streamlit is running and
# provides a file uploader and a dummy "Run" button. Replace with your
# inference wiring when ready.
try:
    import streamlit as st

    # make the app wider and more polished
    st.set_page_config(page_title="AI TraceFinder", layout="wide", initial_sidebar_state="expanded")

    # small CSS polish for cards, header and overall spacing
    st.markdown(
        """
        <style>
        .app-header {display:flex; align-items:center; gap:16px}
        .app-title {font-size:32px; font-weight:700; margin:0}
        .app-sub {color: #6b7280; margin:0}
        .card {background: #ffffff; border-radius:8px; padding:12px; box-shadow: 0 1px 4px rgba(15,23,42,0.06);}
        .uploader-col {padding-right: 16px}
        .results-col {padding-left: 16px}
        .small-muted {color:#6b7280; font-size:12px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    header_col1, header_col2 = st.columns([0.75, 0.25])
    with header_col1:
        st.markdown('<div class="app-header">', unsafe_allow_html=True)
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown('<h1 class="app-title">AI TraceFinder</h1>', unsafe_allow_html=True)
        st.markdown('<p class="app-sub">Quickly inspect scanned images and predict scanner model — diagnostic mode available.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with header_col2:
        # placeholder for model status that will be filled after model loads
        model_status = st.empty()

    # Fixed defaults (removed sidebar controls)
    top_k = 3
    conf_thresh = 0.0
    show_barchart = True
    preproc_option = "ImageNet (default)"
    temp_scale = 1.0
    diagnostic_mode = False

    # Split main area into two columns: left = controls/uploader, right = results
    left_col, right_col = st.columns([0.45, 0.55])

    with left_col:
        st.markdown('<div class="card uploader-col">', unsafe_allow_html=True)
        st.subheader('Upload & Run')
        st.caption('Upload TIFF/JPG/PNG files. Diagnostic mode compares preprocessing options.')
    # show uploader inside card
    uploaded = st.file_uploader("Upload images", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], accept_multiple_files=True, key='uploader')
    run_button = st.button("Run inference")
    st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card results-col">', unsafe_allow_html=True)
        results_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # Inference wiring: load model once and run when user clicks Run.
    @st.cache_resource
    def load_model(path: str = "haar_hybridcnn.pth"):
        device = torch.device("cpu")
        # Try to detect number of classes from the checkpoint so we instantiate
        # the model with the correct output dim. Fall back to 2 if unknown.
        num_classes = 2
        sd = None
        try:
            raw = torch.load(path, map_location=device)
            # raw might be a state_dict (OrderedDict) or a checkpoint dict
            if isinstance(raw, dict) and "state_dict" in raw:
                sd = raw["state_dict"]
            elif isinstance(raw, dict):
                sd = raw
            else:
                sd = None
        except FileNotFoundError:
            st.warning(f"Model checkpoint '{path}' not found. Running with randomly initialized weights.")
            sd = None
        except Exception as e:
            st.error(f"Failed to load model checkpoint: {e}")
            sd = None

        # Infer num_classes from the last linear weight in the classifier if possible
        if sd is not None:
            try:
                # find classifier weight keys
                classifier_weight_keys = [k for k in sd.keys() if "classifier" in k and k.endswith(".weight")]
                if classifier_weight_keys:
                    key = sorted(classifier_weight_keys)[-1]
                    num_classes = int(sd[key].shape[0])
                else:
                    # fallback: pick the last 2D weight tensor (likely final linear)
                    linear_keys = [k for k, v in sd.items() if getattr(v, "ndim", None) == 2 and k.endswith(".weight")]
                    if linear_keys:
                        key = sorted(linear_keys)[-1]
                        num_classes = int(sd[key].shape[0])
                st.info(f"Inferred num_classes={num_classes} from checkpoint.")
            except Exception as e:
                st.warning(f"Could not infer num_classes from checkpoint: {e}")

        model = HybridCNN(num_classes=num_classes)

        if sd is not None:
            # handle DataParallel 'module.' prefixes
            from collections import OrderedDict
            if any(k.startswith("module.") for k in sd.keys()):
                sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())

            res = model.load_state_dict(sd, strict=False)
            # res is a NamedTuple with missing_keys and unexpected_keys
            if getattr(res, "missing_keys", None):
                st.warning(f"Missing keys when loading checkpoint: {res.missing_keys}")
            if getattr(res, "unexpected_keys", None):
                st.warning(f"Unexpected keys when loading checkpoint: {res.unexpected_keys}")

        model.to(device)
        model.eval()
        return model

    model = load_model()

    def preprocess(pil_img, size=(128, 128), option="ImageNet (default)"):
        from torchvision import transforms
        if option == "ImageNet (default)":
            tf = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif option == "No normalization":
            tf = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
            ])
        elif option == "Mean=0.5 Std=0.5":
            tf = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            tf = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
            ])
        return tf(pil_img).unsqueeze(0)

    if run_button:
        if not uploaded:
            st.error("Please upload at least one image first.")
        else:
            import io, os, json
            from PIL import Image
            labels_file = os.path.join(os.path.dirname(__file__), "labels.json")
            if os.path.exists(labels_file):
                try:
                    with open(labels_file, "r", encoding="utf-8") as f:
                        labels_list = json.load(f)
                    if not isinstance(labels_list, list):
                        raise ValueError("labels.json must be a JSON list of strings")
                except Exception as e:
                    st.warning(f"Could not load labels.json: {e}")
                    labels_list = [f"class_{i}" for i in range(getattr(model, 'classifier')[-1].out_features if hasattr(model, 'classifier') else 11)]
            else:
                labels_list = [f"class_{i}" for i in range(getattr(model, 'classifier')[-1].out_features if hasattr(model, 'classifier') else 11)]

            # Ensure labels_list matches model output size: pad with generic names or truncate
            try:
                out_dim = getattr(model, 'classifier')[-1].out_features
            except Exception:
                out_dim = 11
            if len(labels_list) != out_dim:
                st.warning(f"labels.json contains {len(labels_list)} labels but model outputs {out_dim} classes. Padding/truncating labels for display.")
                if len(labels_list) < out_dim:
                    labels_list = labels_list + [f"class_{i}" for i in range(len(labels_list), out_dim)]
                else:
                    labels_list = labels_list[:out_dim]

            results = []
            for file in uploaded:
                try:
                    img = Image.open(io.BytesIO(file.read())).convert('RGB')
                except Exception:
                    # Some TIFFs (big or special tags) may not be readable by PIL; try tifffile if available
                    try:
                        import tifffile
                        arr = tifffile.imread(io.BytesIO(file.read()))
                        from PIL import Image as PILImage
                        img = PILImage.fromarray(arr.astype('uint8')).convert('RGB')
                    except Exception:
                        st.warning(f"Failed to open {getattr(file, 'name', 'uploaded file')} (tried PIL and tifffile)")
                        continue

                st.markdown(f"### {getattr(file, 'name', 'uploaded file')}")
                st.image(img, use_column_width=True)

                # choose preprocessing (or run diagnostics)
                if diagnostic_mode:
                    preproc_options = ["ImageNet (default)", "No normalization", "Mean=0.5 Std=0.5"]
                else:
                    preproc_options = [preproc_option]

                all_reports = []
                for opt in preproc_options:
                    input_tensor = preprocess(img, option=opt)
                    with torch.no_grad():
                        logits = model(input_tensor)
                        if temp_scale != 1.0:
                            logits = logits / float(temp_scale)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                    report = {"preproc": opt, "probs": probs}
                    all_reports.append(report)

                # For normal (non-diagnostic) mode take the first report
                probs = all_reports[0]["probs"]

                # Show diagnostic reports if requested
                if diagnostic_mode:
                    st.write("**Diagnostic reports (per preprocessing option)**")
                    for rep in all_reports:
                        rep_probs = rep['probs']
                        st.write(f"Preprocessing: {rep['preproc']}  — top score: {float(rep_probs.max()):.4f}")
                        idxs = rep_probs.argsort()[::-1][:top_k]
                        for i in idxs:
                            label = labels_list[i] if i < len(labels_list) else f"class_{i}"
                            st.write(f"- {label}: {float(rep_probs[i]):.4f}")
                        if show_barchart:
                            try:
                                import pandas as pd
                                df = pd.DataFrame({"label": [labels_list[i] if i < len(labels_list) else f"class_{i}" for i in range(len(rep_probs))], "score": rep_probs})
                                st.bar_chart(df.sort_values('score', ascending=False).head(top_k).set_index('label'))
                            except Exception as e:
                                st.warning(f"Chart failed for {rep['preproc']}: {e}")
                else:
                    # Get top-k
                    idxs = probs.argsort()[::-1][:top_k]
                    topk = [(labels_list[i] if i < len(labels_list) else f"class_{i}", float(probs[i])) for i in idxs]

                    # Filter by confidence threshold
                    filtered = [t for t in topk if t[1] >= conf_thresh]

                    if filtered:
                        st.write("Top predictions:")
                        for name, score in filtered:
                            st.write(f"- {name}: {score:.3f}")
                    else:
                        st.info("No predictions above the confidence threshold.")

                # Bar chart
                if show_barchart:
                    try:
                        import pandas as pd
                        df = pd.DataFrame({"label": [labels_list[i] if i < len(labels_list) else f"class_{i}" for i in range(len(probs))], "score": probs})
                        df_topk = df.sort_values("score", ascending=False).head(top_k)
                        st.bar_chart(df_topk.set_index('label'))
                    except Exception as e:
                        st.warning(f"Failed to draw bar chart: {e}")

                # Append result row
                predicted_idx = int(probs.argmax())
                predicted_label = labels_list[predicted_idx] if predicted_idx < len(labels_list) else f"class_{predicted_idx}"
                results.append({
                    "file": getattr(file, 'name', 'uploaded file'),
                    "predicted_label": predicted_label,
                    "predicted_score": float(probs[predicted_idx])
                })

            # Show results table and CSV download
            if results:
                try:
                    import pandas as pd, io
                    df_res = pd.DataFrame(results)
                    st.markdown("---")
                    st.write("## Batch results")
                    st.dataframe(df_res)
                    csv = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", data=csv, file_name="results.csv", mime="text/csv")
                except Exception as e:
                    st.warning(f"Failed to present results table: {e}")

    # removed placeholder button

except Exception as e:
    # If Streamlit isn't installed, importing it will raise; we don't want
    # to crash when importing app.py in non-Streamlit contexts.
    print("Streamlit import failed or streamlit app failed to initialize:", e)
