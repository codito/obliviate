// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Obliviate.Ebisu
{
    using System;

    public class EbisuConstraintViolationException : Exception
    {
        public EbisuConstraintViolationException(string message)
            : base(message)
        {
        }
    }
}