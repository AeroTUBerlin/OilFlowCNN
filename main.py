import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from outlier_algorithm import detection_outlier


def load_image(image_path: str, img_res: float, patch_size: int, model_res: int = 23):
    """Load and preprocess an image, resizing and cropping it to match patch dimensions."""
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_raw, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)

    scale = model_res / img_res
    new_shape = tf.cast(tf.shape(img)[:2], tf.float32) * scale
    img_resized = tf.image.resize(img, tf.cast(new_shape, tf.int32), method='lanczos5')

    resized_shape = tf.shape(img_resized)
    n_patches_h = resized_shape[0] // patch_size
    n_patches_w = resized_shape[1] // patch_size

    img_cropped = img_resized[:n_patches_h * patch_size, :n_patches_w * patch_size]
    img_np = img_cropped.numpy()

    patches = extract_patches(img_np, patch_size, n_patches_w.numpy(), n_patches_h.numpy())
    return img_np, patches, n_patches_w.numpy(), n_patches_h.numpy()


def extract_patches(img: np.ndarray, patch_size: int, n_w: int, n_h: int) -> np.ndarray:
    """Split image into non-overlapping patches."""
    patches = img.reshape(n_h, patch_size, n_w, patch_size, -1)
    patches = np.transpose(patches, (0, 2, 1, 3, 4))
    return patches.reshape(n_h * n_w, patch_size, patch_size, -1)


def correction_outlier(y: np.ndarray, radius: float, threshold: float, eps: float,
                       patience: int = 10, max_iter: int = 50):
    """Correct angular outliers via neighborhood averaging."""
    y = np.fmod(y, 2 * np.pi)
    offset_grid = np.array([[-1, 0, 1]] * 3)
    index_offsets = np.stack(np.meshgrid(offset_grid, offset_grid), -1).reshape(-1, 2)
    exclude_center = np.any(index_offsets != 0, axis=1)

    best_quota = 1
    patience_count = 0

    for iteration in range(max_iter):
        U, V = np.cos(y), np.sin(y)
        outliers = detection_outlier(U, V, threshold, radius, eps)
        current_quota = np.mean(outliers)

        print(f"Iter {iteration}: {(1 - current_quota):.2%} inliers, patience {patience_count}")

        if current_quota < best_quota:
            patience_count = 0
        else:
            patience_count += 1
            threshold *= 1.1

        if current_quota <= 1e-4 or patience_count >= patience:
            break

        padded_y = np.pad(y, 1, constant_values=np.nan)
        coords = np.argwhere(outliers)

        for row, col in coords:
            neighbors = index_offsets[exclude_center] + [row + 1, col + 1]
            values = padded_y[neighbors[:, 0], neighbors[:, 1]]
            values = values[~np.isnan(values)]
            if values.size > 0:
                y[row, col] = np.mean(values)

        best_quota = current_quota

    return y, outliers


def predict_and_correct_flow(model, patches: np.ndarray, n_w: int, n_h: int,
                             input_size: int, threshold: float = 1,
                             radius: float = 2, eps: float = 0.5):
    """Predict angular flow from patches and correct outliers."""
    y_pred = model.predict(patches, batch_size=3)
    y_pred = np.reshape(y_pred * np.deg2rad(357) - np.pi, (n_h, n_w))
    y_pred = np.fmod(y_pred, 2 * np.pi)

    y_rotated = y_pred.copy()
    best_outlier_count = 0
    stagnation = 0

    # Flip outliers 180° iteratively
    for _ in range(10):
        U, V = np.cos(y_rotated), np.sin(y_rotated)
        outliers = detection_outlier(U, V, threshold, radius, eps)
        y_rotated[outliers] = np.fmod(y_rotated[outliers] - np.pi, 2 * np.pi)

        if outliers.sum() >= best_outlier_count:
            stagnation += 1
        else:
            stagnation = 0

        if stagnation >= 5:
            break
        best_outlier_count = outliers.sum()

    y_smoothed, outliers_final = correction_outlier(y_rotated.copy(), radius, threshold, eps)
    return y_pred, y_rotated, y_smoothed, outliers, outliers_final


def plot_quiver(image: np.ndarray, U: np.ndarray, V: np.ndarray,
                mask: np.ndarray, patch_size: int, title: str = '', filename: str = ''):
    """Visualize flow field as quiver plot and save as EPS."""
    Y, X = (np.indices(U.shape) + 0.5) * patch_size

    fig, ax = plt.subplots(dpi=300)
    ax.imshow(image, cmap='gray')
    ax.quiver(X, Y, U / 4, -V / 4, mask, angles='xy', pivot='mid', scale_units='xy',
              cmap='bwr_r', edgecolor='k', scale=0.0015, linewidth=0.5,
              headwidth=3, headlength=3.5, headaxislength=3.1, width=0.005)
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()

    if filename:
        plt.savefig(f"{filename}.eps", format='eps', dpi=300)
        plt.close(fig)  # Close to prevent display or memory issues
    else:
        plt.show()

def main():
    # --- Configuration ---
    model_path = './OilFlowCNN.keras'
    image_path = 'test_cases/case_1.png'
    image_resolution = 4  # px/mm

    # --- Load model ---
    model = tf.keras.models.load_model(model_path, compile=False)
    input_size = model.input_shape[1]

    # --- Preprocess image ---
    img, patches, n_w, n_h = load_image(image_path, image_resolution, input_size)

    # --- Predict and correct ---
    y_pred, y_rot, y_corr, outliers_init, outliers_final = predict_and_correct_flow(
        model, patches, n_w, n_h, input_size)

    # --- Convert angles to vector components ---
    U_pred, V_pred = np.cos(y_pred), np.sin(y_pred)
    U_rot, V_rot = np.cos(y_rot), np.sin(y_rot)
    U_corr, V_corr = np.cos(y_corr), np.sin(y_corr)

    # --- Plot results ---
    # plot_quiver(img, U_pred, V_pred, outliers_init, input_size, 'Original Prediction', 'test_cases/original')
    # plot_quiver(img, U_rot, V_rot, outliers_init, input_size, 'After 180° Flip', 'test_cases/outliers_reversed')
    plot_quiver(img, U_corr, V_corr, outliers_final, input_size, 'Smoothed Flow', 'test_cases/final_output')


if __name__ == "__main__":
    main()