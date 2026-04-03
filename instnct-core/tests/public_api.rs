use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("ui")
        .join(name)
}

fn build_temp_crate(case_name: &str, source: &str) -> (PathBuf, PathBuf) {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let sandbox_root = workspace_root
        .join("target")
        .join(format!("uit-{}", std::process::id()));
    let temp_dir = sandbox_root.join(case_name);
    let build_dir = sandbox_root.join(format!("{case_name}-build"));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir).unwrap();
    }
    fs::create_dir_all(temp_dir.join("src")).unwrap();

    let manifest_dir = env!("CARGO_MANIFEST_DIR").replace('\\', "/");
    let cargo_toml = format!(
        "[package]\nname = \"{case_name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n\n[dependencies]\ninstnct-core = {{ path = \"{manifest_dir}\" }}\n"
    );
    fs::write(temp_dir.join("Cargo.toml"), cargo_toml).unwrap();
    fs::write(temp_dir.join("src").join("main.rs"), source).unwrap();

    (temp_dir, build_dir)
}

fn assert_case_passes(case_name: &str, file_name: &str) {
    let source = fs::read_to_string(fixture_path(file_name)).unwrap();
    let (temp_dir, build_dir) = build_temp_crate(case_name, &source);
    let output = Command::new("cargo")
        .arg("check")
        .arg("--manifest-path")
        .arg(temp_dir.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", build_dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "expected {file_name} to compile\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_case_fails(case_name: &str, file_name: &str, expected_fragments: &[&str]) {
    let source = fs::read_to_string(fixture_path(file_name)).unwrap();
    let (temp_dir, build_dir) = build_temp_crate(case_name, &source);
    let output = Command::new("cargo")
        .arg("check")
        .arg("--manifest-path")
        .arg(temp_dir.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", build_dir)
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "expected {file_name} to fail compilation"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    for fragment in expected_fragments {
        assert!(
            stderr.contains(fragment),
            "stderr for {file_name} did not contain `{fragment}`\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn curated_public_surface_is_locked() {
    assert_case_passes("pass_root_surface", "pass_root_surface.rs");
    assert_case_fails(
        "fail_private_module_import",
        "fail_private_module_import.rs",
        &["module `propagation` is private", "PropagationWorkspace"],
    );
    assert_case_fails(
        "fail_unchecked_entrypoint",
        "fail_unchecked_entrypoint.rs",
        &[
            "module `propagation` is private",
            "propagate_token_unchecked",
        ],
    );
    assert_case_fails(
        "fail_build_wave_table",
        "fail_build_wave_table.rs",
        &[
            "no `build_wave_gating_table` in the root",
            "build_wave_gating_table",
        ],
    );
    assert_case_fails(
        "fail_neuron_count_mutation",
        "fail_neuron_count_mutation.rs",
        &[
            "field `neuron_count` of struct `ConnectionGraph` is private",
            "neuron_count",
        ],
    );
}
