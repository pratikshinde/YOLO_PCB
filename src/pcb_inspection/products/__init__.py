"""Product profiles (configs/products/<product_id>/): the only place where product-specific facts live."""

from .profile import (
    ProductProfile,
    ProfileError,
    list_products,
    load_all_profiles,
    load_product_profile,
    products_dir_of,
)

__all__ = ["ProductProfile", "ProfileError", "list_products", "load_all_profiles", "load_product_profile",
           "products_dir_of"]
