fn main() {
    println!("cargo:rerun-if-changed=tauri.conf.json");
    println!("cargo:rerun-if-changed=icons");
    println!("cargo:rerun-if-changed=../INB Surveillance.jpg");
    println!("cargo:rerun-if-changed=../inb_surveillance.jpg");
    tauri_build::build();
}
